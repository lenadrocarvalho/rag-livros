// Marginália — busca híbrida (BM25 + embeddings) 100% no navegador.
// Espelha a lógica de busca.py: mesmos limiares, mesma normalização, mesma fusão RRF.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const MODELO_WEB = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"; // ONNX do mesmo modelo do Python
const URL_REPO = "";  // opcional: "https://github.com/SEU_USUARIO/rag-livros" (senão é deduzido do endereço do GitHub Pages)

const LIMIAR_DISTANCIA = 0.45;
const COBERTURA_MIN = 0.75;
const LIMIAR_DISTANCIA_PALAVRAS = 0.60;
const CANDIDATOS = 30;
const RRF_K = 60;
const RADICAL = 6;
const MAX_RESULTADOS = 8;

const STOPWORDS = `a à ao aos as às até com como da das de dela dele deles do dos e é ela ele eles em entre era
essa esse esta este eu foi for há isso isto já la lhe lo mais mas me mesmo meu minha muito na
nas não nem no nos nós num numa o os ou para pela pelas pelo pelos por qual quando que quem se
sem ser seu sua são só também te tem tinha um uma umas uns vai você estava foram sobre onde
contra porque assim depois antes ainda então`.split(/\s+/);

const EXEMPLOS = [
  "Como eram os olhos de Capitu?",
  "Quem era Bertoleza?",
  "O emplasto de Brás Cubas",
  "Iracema, a virgem dos lábios de mel",
  "Juliana chantageia Luísa",
  "Receita de bolo de chocolate",
];

const $ = (s) => document.querySelector(s);
const semAcento = (s) => s.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
const STOP = new Set(STOPWORDS.map(semAcento));
const PALAVRA = /[\p{L}\p{N}]+/gu;

function termos(texto) {
  const out = [];
  for (const t of semAcento(texto).match(PALAVRA) || []) {
    if (t.length > 1 && !STOP.has(t)) out.push(t.slice(0, RADICAL));
  }
  return out;
}

const estado = {
  livros: [], trechos: [], vetores: null, normas: null, dims: 0,
  tf: [], tam: [], media: 0, idf: new Map(),
  modelo: null, carregandoModelo: null, modo: "palavras", filtro: null, ultimaPergunta: "",
};

/* ---------- armazenamento local (preferências) ---------- */
const prefs = {
  get(k) { try { return localStorage.getItem("marginalia:" + k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem("marginalia:" + k, v); } catch { /* sem storage */ } },
};

/* ---------- carga do acervo ---------- */
async function carregar() {
  const [acervo, bin] = await Promise.all([
    fetch("data/acervo.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    fetch("data/vetores.bin").then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
  ]);
  estado.livros = acervo.livros;
  estado.trechos = acervo.trechos;
  estado.dims = acervo.dims;
  estado.vetores = new Int8Array(bin);
  const n = estado.trechos.length, d = estado.dims;
  estado.normas = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * d, f = j + d; j < f; j++) s += estado.vetores[j] * estado.vetores[j];
    estado.normas[i] = Math.sqrt(s) || 1;
  }
  // índice BM25
  const df = new Map();
  let soma = 0;
  for (const tr of estado.trechos) {
    const tf = new Map();
    for (const t of termos(tr.t)) tf.set(t, (tf.get(t) || 0) + 1);
    let tam = 0; for (const v of tf.values()) tam += v;
    estado.tf.push(tf); estado.tam.push(tam); soma += tam;
    for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
  }
  estado.media = soma / n;
  for (const [t, f] of df) estado.idf.set(t, Math.log(1 + (n - f + 0.5) / (f + 0.5)));
}

/* ---------- busca ---------- */
function rankingPalavras(qs) {
  const k1 = 1.5, b = 0.75, pontos = [];
  for (let i = 0; i < estado.tf.length; i++) {
    if (estado.filtro !== null && estado.trechos[i].l !== estado.filtro) continue;
    const tf = estado.tf[i];
    let s = 0;
    for (const t of qs) {
      const f = tf.get(t);
      if (f) s += estado.idf.get(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * estado.tam[i] / estado.media));
    }
    if (s > 0) pontos.push([s, i]);
  }
  return pontos.sort((a, b) => b[0] - a[0]).slice(0, CANDIDATOS).map((p) => p[1]);
}

function distancia(qv, i) {
  const d = estado.dims, v = estado.vetores;
  let dot = 0;
  for (let j = 0, o = i * d; j < d; j++) dot += qv[j] * v[o + j];
  return 1 - dot / estado.normas[i]; // qv já é normalizado
}

function rankingSentido(qv) {
  const pares = [];
  for (let i = 0; i < estado.trechos.length; i++) {
    if (estado.filtro !== null && estado.trechos[i].l !== estado.filtro) continue;
    pares.push([distancia(qv, i), i]);
  }
  return pares.sort((a, b) => a[0] - b[0]).slice(0, CANDIDATOS).map((p) => p[1]);
}

async function buscar(pergunta) {
  const qs = [...new Set(termos(pergunta))];
  const qv = semanticoAtivo() ? await embutir(pergunta) : null;
  const rSent = qv ? rankingSentido(qv) : [];
  const rPal = qs.length ? rankingPalavras(qs) : [];
  const posSent = new Map(rSent.map((i, r) => [i, r]));
  const posPal = new Map(rPal.map((i, r) => [i, r]));

  const out = [];
  for (const i of new Set([...rSent, ...rPal])) {
    const tf = estado.tf[i];
    const cobertura = qs.length ? qs.filter((t) => tf.has(t)).length / qs.length : 0;
    const dist = qv ? distancia(qv, i) : null;
    const aceito = qv
      ? dist <= LIMIAR_DISTANCIA
        || (cobertura >= COBERTURA_MIN && dist <= LIMIAR_DISTANCIA_PALAVRAS)
        || (cobertura === 1 && qs.length >= 2)
      // só palavras: exige que o trecho contenha os termos da pergunta
      : (cobertura === 1) || (cobertura >= COBERTURA_MIN && qs.length >= 2);
    if (!aceito) continue;
    let score = 0;
    if (posSent.has(i)) score += 1 / (RRF_K + posSent.get(i));
    if (posPal.has(i)) score += 1 / (RRF_K + posPal.get(i));
    out.push({ i, dist, cobertura, score });
  }
  return { qs, resultados: out.sort((a, b) => b.score - a.score).slice(0, MAX_RESULTADOS) };
}

/* ---------- modelo semântico (opcional) ---------- */
const semanticoAtivo = () => estado.modo === "sentido" && estado.modelo !== null;

async function embutir(texto) {
  const r = await estado.modelo(texto, { pooling: "mean", normalize: true });
  return r.data;
}

function carregarModelo() {
  if (estado.modelo) return Promise.resolve();
  if (estado.carregandoModelo) return estado.carregandoModelo;
  const status = $("#modo-status"), prog = $("#progresso"), barra = $("#progresso-barra");
  status.textContent = "Baixando o modelo de linguagem (118 MB, só na primeira vez)…";
  prog.hidden = false;
  const arquivos = new Map();
  estado.carregandoModelo = (async () => {
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    estado.modelo = await pipeline("feature-extraction", MODELO_WEB, {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          arquivos.set(p.file, [p.loaded, p.total]);
          let l = 0, t = 0; for (const [a, b] of arquivos.values()) { l += a; t += b; }
          barra.style.width = `${(100 * l / t).toFixed(1)}%`;
          status.textContent = `Baixando o modelo… ${Math.round(l / 1e6)} de ${Math.round(t / 1e6)} MB`;
        }
      },
    });
    prog.hidden = true;
    if (estado.modo === "sentido") status.textContent = "Busca por sentido ativa. Entende paráfrases, não só palavras iguais.";
  })().catch((e) => {
    console.error(e);
    estado.carregandoModelo = null;
    prog.hidden = true;
    definirModo("palavras", false);
    status.textContent = "Não foi possível carregar o modelo. Continuando com a busca por palavras.";
  });
  return estado.carregandoModelo;
}

async function definirModo(modo, rebuscar = true) {
  estado.modo = modo;
  prefs.set("modo", modo);
  document.querySelectorAll(".modo button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.modo === modo)));
  if (modo === "sentido") {
    await carregarModelo();
  } else {
    $("#modo-status").textContent = "Busca por palavras: rápida e literal. Ative “Sentido” para paráfrases.";
  }
  if (rebuscar && estado.ultimaPergunta) executar(estado.ultimaPergunta);
}

/* ---------- renderização ---------- */
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cor = (l) => `var(--l${l % 6})`;

// O PDF quebra linhas no meio dos parágrafos; refaz o fluxo do texto.
// Mantém quebras antes de falas (—) e de linhas em CAIXA ALTA (títulos de capítulo).
function fluir(texto, { aparar = true } = {}) {
  const linhas = texto.split(/\n/);
  let s = linhas[0];
  for (let k = 1; k < linhas.length; k++) {
    const l = linhas[k].trim();
    if (!l) { s += "\n"; continue; }
    const quebra = /^[—–-]/.test(l) || (l.length < 60 && l === l.toUpperCase() && /\p{Lu}/u.test(l))
      || /^CAP[IÍ]TULO\b/i.test(l) || /^[IVXLC]+\.?$/.test(l);
    s += (quebra || s.endsWith("\n") ? "\n" : " ") + l;
  }
  s = s.replace(/\n{3,}/g, "\n\n").trim();
  if (aparar) {
    // trechos se sobrepõem: se começa no meio de uma frase, corta a palavra parcial
    if (/^\p{Ll}/u.test(s)) s = "… " + s.replace(/^\S+\s+/, "");
    if (!/[.!?…:»”"]$/.test(s)) s = s.replace(/\s+\S*$/, "") + " …";
  }
  return s;
}

function frases(texto) {
  return texto.match(/[^.!?…]*[.!?…]+["”»]?\s*|[^.!?…]+$/g) || [texto];
}

function realcar(texto, qs) {
  const alvo = new Set(qs);
  const partes = frases(texto);
  let melhor = -1, melhorN = 0;
  partes.forEach((f, k) => {
    const n = new Set(termos(f).filter((t) => alvo.has(t))).size;
    if (n > melhorN) { melhorN = n; melhor = k; }
  });
  const marcarTermos = (s) => s.split(/([\p{L}\p{N}]+)/u).map((p, k) => {
    if (k % 2 === 0) return esc(p);
    const t = semAcento(p).slice(0, RADICAL);
    return p.length > 1 && alvo.has(t) && !STOP.has(semAcento(p)) ? `<span class="termo">${esc(p)}</span>` : esc(p);
  }).join("");
  const html = partes.map((f, k) => k === melhor ? `<span class="frase">${marcarTermos(f)}</span>` : marcarTermos(f)).join("");
  return { html, frase: (melhor >= 0 ? partes[melhor] : partes[0]).trim() };
}

const nomeLivro = (l) => estado.livros[l].titulo;

function renderEstante() {
  const el = $("#livros");
  el.innerHTML = "";
  const total = estado.trechos.length;
  $("#estante-resumo").textContent = `${estado.livros.length} livros · ${total.toLocaleString("pt-BR")} trechos · toque para filtrar`;
  estado.livros.forEach((b, l) => {
    const bt = document.createElement("button");
    bt.type = "button";
    bt.className = "livro";
    bt.style.setProperty("--cor", cor(l));
    bt.setAttribute("aria-pressed", String(estado.filtro === l));
    bt.innerHTML = `<span class="livro__ano">${b.ano ?? ""}</span>
      <span class="livro__titulo">${esc(b.titulo)}</span>
      <span class="livro__autor">${esc(b.autor || "")}</span>
      <span class="livro__meta">${b.paginas} págs. · ${b.trechos} trechos</span>`;
    bt.addEventListener("click", () => {
      estado.filtro = estado.filtro === l ? null : l;
      renderEstante();
      if (estado.ultimaPergunta) executar(estado.ultimaPergunta);
      else $("#pergunta").focus();
    });
    el.appendChild(bt);
  });
}

function renderVeredito(res, qs) {
  const v = $("#veredito");
  const filtro = estado.filtro !== null ? ` em <i>${esc(nomeLivro(estado.filtro))}</i>` : "";
  if (!res.length) {
    v.className = "veredito veredito--vazio";
    const dicas = [];
    if (!semanticoAtivo()) dicas.push("ative a busca por <b>Sentido</b> para encontrar paráfrases");
    if (estado.filtro !== null) dicas.push("remova o filtro de livro");
    dicas.push("use nomes de personagens ou palavras que estariam no texto");
    v.innerHTML = `<span class="veredito__selo"><svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.01"/><circle cx="12" cy="12" r="9"/></svg></span>
      <div><h2>Os livros não dizem isso${filtro}.</h2>
      <p>Nenhum trecho passou no filtro de evidência. Em vez de inventar uma resposta, a Marginália prefere não responder.</p>
      <ul>${dicas.map((d) => `<li>${d}</li>`).join("")}</ul></div>`;
    return;
  }
  const nLivros = new Set(res.map((r) => estado.trechos[r.i].l)).size;
  const via = semanticoAtivo() ? "por sentido e por palavras" : "por palavras";
  v.className = "veredito";
  v.innerHTML = `<span class="veredito__selo"><svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg></span>
    <div><h2>${res.length} ${res.length === 1 ? "trecho encontrado" : "trechos encontrados"} em ${nLivros} ${nLivros === 1 ? "livro" : "livros"}${filtro}.</h2>
    <p>Busca ${via}${qs.length ? ` · termos: ${qs.map((t) => `<code>${esc(t)}</code>`).join(" ")}` : ""}. Leia as citações: a resposta está no texto, não em um resumo.</p></div>`;
}

function renderMapa(res) {
  const m = $("#mapa");
  m.innerHTML = "";
  if (!res.length) return;
  const porLivro = new Map();
  res.forEach((r, k) => {
    const l = estado.trechos[r.i].l;
    if (!porLivro.has(l)) porLivro.set(l, []);
    porLivro.get(l).push([r, k]);
  });
  for (const [l, itens] of [...porLivro].sort((a, b) => a[0] - b[0])) {
    const b = estado.livros[l];
    const linha = document.createElement("div");
    linha.className = "mapa__linha";
    linha.style.setProperty("--cor", cor(l));
    linha.innerHTML = `<span class="mapa__nome" title="${esc(b.titulo)}">${esc(b.titulo)}</span><div class="mapa__trilho"></div>`;
    const trilho = linha.querySelector(".mapa__trilho");
    for (const [r, k] of itens) {
      const p = estado.trechos[r.i].p;
      const mk = document.createElement("button");
      mk.type = "button";
      mk.className = "mapa__marca";
      mk.style.left = `${Math.min(99, Math.max(1, (100 * (p - 0.5)) / Math.max(b.paginas, 1)))}%`;
      mk.style.setProperty("--forca", (1 - k / (res.length + 2)).toFixed(2));
      mk.title = `${b.titulo}, p. ${p} — resultado ${k + 1}`;
      mk.setAttribute("aria-label", mk.title);
      mk.addEventListener("click", () => {
        const c = document.getElementById(`r${k}`);
        c.scrollIntoView({ block: "start" });
        c.classList.remove("cartao--destaque"); void c.offsetWidth; c.classList.add("cartao--destaque");
      });
      trilho.appendChild(mk);
    }
    m.appendChild(linha);
  }
  const leg = document.createElement("p");
  leg.className = "mapa__legenda";
  leg.textContent = "Onde cada trecho aparece no livro, do início ao fim. Marcas mais fortes = mais relevantes.";
  m.appendChild(leg);
}

function sinal(rotulo, valor) {
  const pct = Math.round(Math.max(0, Math.min(1, valor)) * 100);
  return `<span class="sinal" title="${rotulo}: ${pct}%">${rotulo}<span class="sinal__barra"><i style="width:${pct}%"></i></span>${pct}%</span>`;
}

function vizinho(i, passo) {
  const j = i + passo;
  const t = estado.trechos[j];
  return t && t.l === estado.trechos[i].l ? t : null;
}

function renderCartoes(res, qs) {
  const ol = $("#cartoes");
  ol.innerHTML = "";
  const tpl = $("#tpl-cartao");
  res.forEach((r, k) => {
    const tr = estado.trechos[r.i];
    const b = estado.livros[tr.l];
    const li = tpl.content.firstElementChild.cloneNode(true);
    li.id = `r${k}`;
    li.style.setProperty("--cor", cor(tr.l));
    li.querySelector(".cartao__livro").innerHTML = `${esc(b.titulo)} <small>· ${esc(b.autor || "")}${b.ano ? `, ${b.ano}` : ""}</small>`;
    li.querySelector(".cartao__pagina").textContent = `p. ${tr.p}`;
    const { html, frase } = realcar(fluir(tr.t), qs);
    li.querySelector(".cartao__texto").innerHTML = html;
    const sinais = [];
    if (r.dist !== null) sinais.push(sinal("sentido", 1 - r.dist));
    if (qs.length) sinais.push(sinal("palavras", r.cobertura));
    li.querySelector(".cartao__sinais").innerHTML = sinais.join("");

    const antes = li.querySelector(".cartao__contexto--antes");
    const depois = li.querySelector(".cartao__contexto--depois");
    const btCtx = li.querySelector('[data-acao="contexto"]');
    const ant = vizinho(r.i, -1), prox = vizinho(r.i, 1);
    if (!ant && !prox) btCtx.hidden = true;
    btCtx.addEventListener("click", () => {
      const abrir = antes.hidden && depois.hidden;
      if (ant) { antes.textContent = "… " + fluir(ant.t, { aparar: false }).slice(-450).replace(/^\S*\s/, ""); antes.hidden = !abrir; }
      if (prox) { depois.textContent = fluir(prox.t, { aparar: false }).slice(0, 450).replace(/\s\S*$/, "") + " …"; depois.hidden = !abrir; }
      btCtx.setAttribute("aria-expanded", String(abrir));
      btCtx.textContent = abrir ? "Recolher" : "Ler ao redor";
    });
    li.querySelector('[data-acao="citar"]').addEventListener("click", async () => {
      const cit = `“${frase.replace(/\s+/g, " ")}” — ${b.autor ? b.autor + ", " : ""}${b.titulo}${b.ano ? ` (${b.ano})` : ""}, p. ${tr.p}.`;
      try { await navigator.clipboard.writeText(cit); aviso("Citação copiada"); }
      catch { aviso("Não foi possível copiar"); }
    });
    ol.appendChild(li);
  });
}

let timerAviso;
function aviso(txt) {
  const a = $("#aviso");
  a.textContent = txt;
  a.classList.add("aviso--on");
  clearTimeout(timerAviso);
  timerAviso = setTimeout(() => a.classList.remove("aviso--on"), 1800);
}

async function executar(pergunta) {
  pergunta = pergunta.trim();
  if (!pergunta) return;
  estado.ultimaPergunta = pergunta;
  $("#pergunta").value = pergunta;
  const url = new URL(location.href);
  url.searchParams.set("q", pergunta);
  if (estado.filtro !== null) url.searchParams.set("livro", estado.livros[estado.filtro].id); else url.searchParams.delete("livro");
  history.replaceState(null, "", url);

  if (estado.modo === "sentido" && !estado.modelo) await carregarModelo();
  const { qs, resultados } = await buscar(pergunta);
  if (pergunta !== estado.ultimaPergunta) return; // outra busca começou nesse meio-tempo
  $("#resultados").hidden = false;
  $("#estante").classList.add("estante--compacta");
  renderVeredito(resultados, qs);
  renderMapa(resultados);
  renderCartoes(resultados, qs);
  if (window.matchMedia("(max-width: 600px)").matches) $("#resultados").scrollIntoView({ block: "start" });
}

/* ---------- início ---------- */
function iniciarTema() {
  const salvo = prefs.get("tema");
  if (salvo) document.documentElement.dataset.theme = salvo;
  $("#tema").addEventListener("click", () => {
    const escuro = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const novo = escuro ? "light" : "dark";
    document.documentElement.dataset.theme = novo;
    prefs.set("tema", novo);
  });
}

function iniciarLinkRepo() {
  const a = $("#link-github");
  let url = URL_REPO;
  const m = location.hostname.match(/^([\w-]+)\.github\.io$/);
  if (!url && m) url = `https://github.com/${m[1]}/${location.pathname.split("/").filter(Boolean)[0] || ""}`;
  if (url) a.href = url; else a.hidden = true;
}

async function main() {
  iniciarTema();
  iniciarLinkRepo();
  const ex = $("#exemplos");
  for (const q of EXEMPLOS) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = q;
    b.addEventListener("click", () => executar(q));
    ex.appendChild(b);
  }
  $("#livros").innerHTML = '<div class="esqueleto"></div>'.repeat(5);
  try {
    await carregar();
  } catch (e) {
    console.error(e);
    $("#estante-resumo").textContent = "";
    $("#livros").innerHTML = `<p>Não consegui abrir o acervo (<code>data/acervo.json</code>). Se abriu o arquivo direto do disco,
      sirva a pasta: <code>python -m http.server -d docs</code> e acesse <code>http://localhost:8000</code>.</p>`;
    return;
  }
  const params = new URLSearchParams(location.search);
  const livroParam = params.get("livro");
  if (livroParam) {
    const l = estado.livros.findIndex((b) => b.id === livroParam);
    if (l >= 0) estado.filtro = l;
  }
  renderEstante();

  const input = $("#pergunta");
  input.disabled = false;
  document.querySelector(".busca__ir").disabled = false;
  $("#form").addEventListener("submit", (e) => { e.preventDefault(); executar(input.value); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== input && !e.ctrlKey && !e.metaKey) { e.preventDefault(); input.focus(); }
  });
  document.querySelectorAll(".modo button").forEach((b) => b.addEventListener("click", () => definirModo(b.dataset.modo)));

  const modoSalvo = prefs.get("modo") === "sentido" ? "sentido" : "palavras";
  const q = params.get("q");
  definirModo(modoSalvo, false); // carrega o modelo em segundo plano, se for o caso
  if (q) executar(q); else if (matchMedia("(min-width: 601px)").matches) input.focus();
}

main();
