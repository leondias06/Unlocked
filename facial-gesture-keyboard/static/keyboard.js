// Facial Gesture Keyboard - the on-screen keyboard itself.
//
// Runs in its own window/page, separate from the camera+calibration UI
// (see index.html / app.js), so it can be a translucent, always-on-top
// overlay that floats over whatever app the user is actually typing
// into. It does not do gesture recognition itself - that happens in
// the main window, which has the camera; this page just renders the
// grid and reacts to gestures forwarded from Python (desktop_app.py
// evaluates `applyGestureToKeyboard(label)` in this window whenever a
// gesture fires in the main window), plus physical keys for testing.
//
// Layout is an 11-column x 7-row grid matching the reference design:
// row 0 is esc + 5 predictive-text cells, rows 1-5 are the alphabet
// (plain A-Z reading order, not QWERTY) plus a numbers/symbols column
// and the function keys (tab/caps/enter/backspace), row 6 is the
// bottom function row (volume/brightness + blank custom keys).
//
// Cells that span multiple rows/columns (caps, enter, space, etc.) are
// single DOM elements placed via CSS grid-row/grid-column spans; a
// separate occupancy map resolves "what cell is at this grid position"
// for keyboard/gesture navigation, so moving across a spanned key just
// works without special-casing.

const GRID_COLS = 11;
const GRID_ROWS = 7;

// row/col are 0-indexed top-left anchor; rs/cs default to 1.
// kind determines both rendering and what confirming the key does:
//   char       - primary is the letter; case follows capsOn
//   shiftchar  - primary when capsOn is off, shifted when on
//   space / tab / esc / enter / backspace - real keystrokes
//   caps       - toggles capsOn (a local state, not the OS caps lock -
//                see confirmKey())
//   volume_up/down    - real OS media keys
//   brightness_up/down - best-effort real OS brightness (WMI; not all
//                displays support it - silently does nothing if not)
//   suggestion - predictive-text placeholder (wired up in a later pass)
//   mousetoggle - switches to mouse-cursor mode (not implemented yet)
//   poweronoff / blank - placeholders, deliberately do nothing
const CELLS = [
  // row 0: esc + 5 predictive-text suggestion cells
  { row: 0, col: 0, kind: "esc", label: "esc" },
  { row: 0, col: 1, cs: 2, kind: "suggestion", idx: 0 },
  { row: 0, col: 3, cs: 2, kind: "suggestion", idx: 1 },
  { row: 0, col: 5, cs: 2, kind: "suggestion", idx: 2 },
  { row: 0, col: 7, cs: 2, kind: "suggestion", idx: 3 },
  { row: 0, col: 9, cs: 2, kind: "suggestion", idx: 4 },

  // row 1
  { row: 1, col: 0, kind: "char", primary: "a" },
  { row: 1, col: 1, kind: "char", primary: "b" },
  { row: 1, col: 2, kind: "char", primary: "c" },
  { row: 1, col: 3, kind: "char", primary: "d" },
  { row: 1, col: 4, kind: "char", primary: "e" },
  { row: 1, col: 5, kind: "char", primary: "f" },
  { row: 1, col: 6, kind: "tab", label: "tab" },
  { row: 1, col: 7, kind: "shiftchar", primary: "1", shifted: "!" },
  { row: 1, col: 8, kind: "shiftchar", primary: "2", shifted: "@" },
  { row: 1, col: 9, kind: "shiftchar", primary: "3", shifted: "#" },
  { row: 1, col: 10, kind: "shiftchar", primary: "-", shifted: "_" },

  // row 2
  { row: 2, col: 0, kind: "char", primary: "g" },
  { row: 2, col: 1, kind: "char", primary: "h" },
  { row: 2, col: 2, kind: "char", primary: "i" },
  { row: 2, col: 3, kind: "char", primary: "j" },
  { row: 2, col: 4, kind: "char", primary: "k" },
  { row: 2, col: 5, kind: "char", primary: "l" },
  { row: 2, col: 6, rs: 2, kind: "caps", label: "caps" },
  { row: 2, col: 7, kind: "shiftchar", primary: "4", shifted: "$" },
  { row: 2, col: 8, kind: "shiftchar", primary: "5", shifted: "%" },
  { row: 2, col: 9, kind: "shiftchar", primary: "6", shifted: "^" },
  { row: 2, col: 10, kind: "shiftchar", primary: "=", shifted: "+" },

  // row 3 (caps continues from row 2 at col 6)
  { row: 3, col: 0, kind: "char", primary: "m" },
  { row: 3, col: 1, kind: "char", primary: "n" },
  { row: 3, col: 2, kind: "char", primary: "o" },
  { row: 3, col: 3, kind: "char", primary: "p" },
  { row: 3, col: 4, kind: "char", primary: "q" },
  { row: 3, col: 5, kind: "char", primary: "r" },
  { row: 3, col: 7, kind: "shiftchar", primary: "7", shifted: "&" },
  { row: 3, col: 8, kind: "shiftchar", primary: "8", shifted: "*" },
  { row: 3, col: 9, kind: "shiftchar", primary: "9", shifted: "(" },
  { row: 3, col: 10, kind: "shiftchar", primary: "<", shifted: ">" },

  // row 4
  { row: 4, col: 0, kind: "char", primary: "s" },
  { row: 4, col: 1, kind: "char", primary: "t" },
  { row: 4, col: 2, kind: "char", primary: "u" },
  { row: 4, col: 3, kind: "char", primary: "v" },
  { row: 4, col: 4, kind: "char", primary: "w" },
  { row: 4, col: 5, kind: "char", primary: "x" },
  { row: 4, col: 6, rs: 2, kind: "enter", label: "enter" },
  { row: 4, col: 7, kind: "shiftchar", primary: "0", shifted: ")" },
  { row: 4, col: 8, kind: "shiftchar", primary: "'", shifted: '"' },
  { row: 4, col: 9, cs: 2, kind: "mousetoggle", label: "toggle" },

  // row 5 (enter continues from row 4 at col 6)
  { row: 5, col: 0, kind: "char", primary: "y" },
  { row: 5, col: 1, kind: "char", primary: "z" },
  { row: 5, col: 2, cs: 2, kind: "space", label: "␣" },
  { row: 5, col: 4, cs: 2, kind: "backspace", label: "backspace" },
  { row: 5, col: 7, kind: "shiftchar", primary: ".", shifted: ":" },
  { row: 5, col: 8, kind: "shiftchar", primary: ",", shifted: "?" },
  { row: 5, col: 9, cs: 2, kind: "poweronoff", label: "on | off" },

  // row 6: bottom function row - volume/brightness are real; the rest
  // are deliberately blank custom keys (navigable/selectable, do
  // nothing - no setup system for these, per spec)
  { row: 6, col: 0, kind: "blank" },
  { row: 6, col: 1, kind: "blank" },
  { row: 6, col: 2, kind: "blank" },
  { row: 6, col: 3, kind: "volume_down", label: "\u{1F509}" },
  { row: 6, col: 4, kind: "volume_up", label: "\u{1F50A}" },
  { row: 6, col: 5, kind: "brightness_down", label: "☀︎-" },
  { row: 6, col: 6, kind: "brightness_up", label: "☀︎+" },
  { row: 6, col: 7, kind: "blank" },
  { row: 6, col: 8, kind: "blank" },
  { row: 6, col: 9, kind: "blank" },
  { row: 6, col: 10, kind: "blank" },
];

const FUNCTION_KINDS = new Set([
  "esc", "tab", "caps", "enter", "backspace", "mousetoggle", "poweronoff",
]);

// Predictive text word pool, most-common-first. The first ~150 or so
// (the/of/and/a/to/...) are the classic high-frequency English function
// words, reliably in that order; past that it widens into common
// everyday vocabulary that isn't strictly frequency-ranked, just there
// so prefix matching has something reasonable to find for more words
// than the core set covers. No network calls - bundled directly so
// suggestions work fully offline, same as everything else in this app.
const COMMON_WORDS = [
  "the","of","and","a","to","in","is","you","that","it","he","was","for","on","are","as","with",
  "his","they","i","at","be","this","have","from","or","one","had","by","word","but","not","what",
  "all","were","we","when","your","can","said","there","use","an","each","which","she","do","how",
  "their","if","will","up","other","about","out","many","then","them","these","so","some","her",
  "would","make","like","him","into","time","has","look","two","more","write","go","see","number",
  "no","way","could","people","my","than","first","water","been","call","who","its","now","find",
  "long","down","day","did","get","come","made","may","part","over","new","sound","take","only",
  "little","work","know","place","year","live","me","back","give","most","very","after","thing",
  "our","just","name","good","man","think","say","great","where","help","through","much","before",
  "line","right","too","mean","old","any","same","tell","boy","follow","came","want","show","around",
  "form","three","small","set","put","end","does","another","well","large","must","big","even",
  "such","because","turn","here","why","ask","went","men","read","need","land","different","home",
  "us","move","try","kind","hand","picture","again","change","off","play","spell","air","away",
  "animal","house","point","page","letter","mother","answer","found","study","still","learn",
  "should","america","world","high","every","near","add","food","between","own","below","country",
  "plant","last","school","father","keep","tree","never","start","city","earth","eye","light",
  "thought","head","under","story","saw","left","don't","few","while","along","might","close",
  "something","seem","next","hard","open","example","begin","life","always","those","both","paper",
  "together","got","group","often","run","important","until","children","side","feet","car","mile",
  "night","walk","white","sea","began","grow","took","river","four","carry","state","once","book",
  "hear","stop","without","second","later","miss","idea","enough","eat","face","watch","far","indian",
  "really","almost","let","above","girl","sometimes","mountain","cut","young","talk","soon","list",
  "song","being","leave","family","it's",
  // Everyday-needs vocabulary bumped ahead of the broader word pool below -
  // this app is built for people who may only be able to communicate
  // through this keyboard, so words like "hungry" or "bathroom" matter
  // more here than raw corpus frequency would rank them.
  "hello","hi","yes","please","thanks","thank","sorry","hungry","thirsty","bathroom","toilet",
  "uncomfortable","stop","okay","nurse","medicine","pillow","blanket","hurt",
  "mom","dad","mother","grandma","grandpa","wife","son","daughter",
  "happy","sad","angry","comfortable","warm","cool","quiet","loud","easy","difficult",
  "wonderful","terrible","amazing","perfect","ready","busy","free","tired","hurts",
  "running","walking","talking","working","playing","looking","going","coming","wanting",
  "needing","feeling","thinking","saying","asking","telling","calling","trying","helping",
  "loving","giving","taking","making","getting","keeping","holding","bringing","showing",
  "starting","stopping","opening","closing","turning","moving","sitting","standing",
  "sleeping","eating","drinking","watching","listening","reading","writing","thanking",
  "one","two","three","four","five","six","seven","eight","nine","ten","twenty","thirty",
  "about","above","across","act","active","actually","add","address","admit","adult","affect",
  "afford","afraid","after","again","against","age","agency","agent","ago","agree","ahead","air",
  "allow","almost","alone","already","also","although","always","among","amount","analysis",
  "announce","another","any","anyone","anything","appear","apply","approach","area","argue","arm",
  "arrive","art","article","artist","assume","attack","attention","attorney","audience","author",
  "authority","available","avoid","baby","bad","bag","ball","bank","bar","base","beat","beautiful",
  "become","bed","begin","behavior","behind","believe","benefit","best","better","between","beyond",
  "bill","billion","bit","black","blood","blue","board","body","born","both","box","break","bring",
  "brother","budget","build","building","business","buy","camera","campaign","cancer","candidate",
  "capital","card","care","career","case","catch","cause","cell","center","central","century",
  "certain","certainly","chair","challenge","chance","character","charge","check","child","choice",
  "choose","church","citizen","civil","claim","class","clear","clearly","close","coach","cold",
  "collection","college","color","commercial","common","community","company","compare","computer",
  "concern","condition","conference","congress","consider","consumer","contain","continue","control",
  "cost","couple","course","court","cover","create","crime","cultural","culture","cup","current",
  "customer","cut","dark","data","daughter","dead","deal","death","debate","decade","decide",
  "decision","deep","defense","degree","democrat","describe","design","despite","detail","determine",
  "develop","development","die","difference","difficult","dinner","direction","director","discover",
  "discuss","discussion","disease","doctor","dog","door","draw","dream","drive","drop","drug","east",
  "easy","economic","economy","edge","education","effect","effort","either","election","else",
  "employee","energy","enjoy","enter","entire","environment","especially","establish","evening",
  "event","ever","everybody","everyone","everything","evidence","exactly","example","executive",
  "exist","expect","experience","expert","explain","face","fact","factor","fail","fall","family",
  "fast","father","fear","federal","feel","feeling","field","fight","figure","fill","film","final",
  "finally","financial","fine","finger","finish","fire","firm","fish","fly","focus","food","foot",
  "force","foreign","forget","former","forward","free","friend","front","full","fund","future",
  "game","garden","gas","general","generation","girl","glass","goal","government","green","ground",
  "growth","gun","guy","hair","half","happen","happy","hard","head","health","heart","heat","heavy",
  "hold","hope","hospital","hot","hotel","hour","huge","human","hundred","husband","identify","image",
  "imagine","impact","important","improve","include","including","increase","indeed","indicate",
  "individual","industry","information","inside","instead","institution","interest","interesting",
  "international","interview","investment","involve","issue","item","itself","job","join","key",
  "kid","kitchen","knowledge","language","late","later","laugh","law","lawyer","lead","leader",
  "least","leg","legal","less","level","lie","local","lose","loss","lot","love","low","machine",
  "magazine","main","maintain","major","majority","manage","management","manager","market","marriage",
  "material","matter","maybe","measure","media","medical","meet","meeting","member","memory",
  "mention","message","method","middle","military","million","mind","minute","mission","model",
  "modern","moment","money","month","morning","movement","movie","music","myself","nation","national",
  "natural","nature","news","newspaper","nice","north","note","office","officer","official","often",
  "operation","opportunity","option","order","organization","outside","owner","pain","painting",
  "parent","particular","particularly","partner","party","pass","past","patient","pattern","pay",
  "peace","perform","performance","perhaps","period","person","personal","phone","physical","pick",
  "plan","player","police","policy","political","politics","poor","popular","population","position",
  "positive","possible","power","practice","prepare","present","president","pressure","pretty",
  "prevent","price","private","probably","problem","process","produce","product","production",
  "professional","professor","program","project","property","protect","prove","provide","public",
  "pull","purpose","push","quality","question","quickly","quite","race","radio","raise","range",
  "rate","rather","reach","ready","real","reality","realize","reason","receive","recent","recognize",
  "record","reduce","reflect","region","relate","relationship","religious","remain","remember",
  "remove","report","represent","require","research","resource","respond","response","responsibility",
  "rest","result","return","reveal","rich","rise","risk","road","rock","role","room","rule","safe",
  "save","scene","science","scientist","score","season","seat","section","security","seek","sell",
  "send","senior","sense","series","serious","serve","service","sex","sexual","shake","share",
  "shoot","short","shot","shoulder","sign","significant","similar","simple","simply","sing","single",
  "sister","site","situation","size","skill","skin","social","society","soldier","somebody","son",
  "sort","source","south","southern","space","speak","special","specific","speech","spend","sport",
  "spring","staff","stage","stand","standard","star","state","statement","station","stay","step",
  "stock","store","strategy","street","strong","structure","student","stuff","style","subject",
  "success","successful","suddenly","suffer","suggest","summer","support","surface","system","table",
  "talk","task","tax","teach","teacher","team","technology","television","tend","term","test",
  "theory","threat","throw","total","tough","toward","town","trade","traditional","training","travel",
  "treat","treatment","trial","trip","trouble","true","truth","type","understand","unit","various",
  "victim","view","violence","visit","voice","vote","wait","wall","war","weapon","wear","week",
  "weight","western","whatever","whether","whole","wide","wife","win","wind","window","wish",
  "within","without","woman","wonder","worker","worry","writer","wrong","yard","yeah","yet","young",
];

const kbOutput = document.getElementById("kbOutput");
const kbGrid = document.getElementById("kbGrid");

let curRow = 0;
let curCol = 0;
let typedText = "";
let capsOn = false;
let ws = null;
let currentSuggestions = [];

// occupancy[row][col] -> index into CELLS
const occupancy = Array.from({ length: GRID_ROWS }, () => new Array(GRID_COLS).fill(-1));
CELLS.forEach((cell, i) => {
  const rs = cell.rs || 1;
  const cs = cell.cs || 1;
  for (let r = cell.row; r < cell.row + rs; r++) {
    for (let c = cell.col; c < cell.col + cs; c++) {
      occupancy[r][c] = i;
    }
  }
});

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onclose = () => setTimeout(connectWS, 1500);
  ws.onerror = () => ws.close();
}

function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(obj));
}

function labelFor(cell) {
  if (cell.kind === "char") return capsOn ? cell.primary.toUpperCase() : cell.primary;
  if (cell.kind === "shiftchar") return capsOn ? cell.shifted : cell.primary;
  if (cell.kind === "suggestion") return ""; // filled in by updateSuggestions()
  return cell.label;
}

function buildKeyboardGrid() {
  kbGrid.innerHTML = "";
  CELLS.forEach((cell, i) => {
    const el = document.createElement("div");
    el.className = "kb-key";
    if (FUNCTION_KINDS.has(cell.kind)) el.classList.add("kb-key--function");
    if (cell.kind === "suggestion") el.classList.add("kb-key--suggestion");
    if (cell.kind === "blank") el.classList.add("kb-key--blank");
    if (cell.kind === "volume_up" || cell.kind === "volume_down" ||
        cell.kind === "brightness_up" || cell.kind === "brightness_down") {
      el.classList.add("kb-key--media");
    }
    el.style.gridColumn = `${cell.col + 1} / span ${cell.cs || 1}`;
    el.style.gridRow = `${cell.row + 1} / span ${cell.rs || 1}`;
    el.textContent = labelFor(cell);
    el.addEventListener("click", () => {
      curRow = cell.row;
      curCol = cell.col;
      renderCursor();
      confirmKey();
    });
    cell.el = el;
    kbGrid.appendChild(el);
  });
  renderCursor();
}

function refreshLabels() {
  for (const cell of CELLS) {
    if (cell.kind === "char" || cell.kind === "shiftchar") {
      cell.el.textContent = labelFor(cell);
    }
  }
}

function currentCell() {
  return CELLS[occupancy[curRow][curCol]];
}

function renderCursor() {
  for (const cell of CELLS) cell.el.classList.remove("is-cursor");
  currentCell().el.classList.add("is-cursor");
}

function renderOutput() {
  // A real <textarea>, not just a styled preview - the typed text is
  // genuinely usable (select, copy, paste elsewhere), with no
  // dependency on OS-level access. Separately, confirmKey() below also
  // sends real OS keystrokes via the server for typing into whatever
  // window currently has focus (Word, a browser, etc).
  kbOutput.value = typedText;
  kbOutput.scrollTop = kbOutput.scrollHeight;
  updateSuggestions();
}

// The word currently being typed: the trailing run of letters/apostrophes
// since the last space/newline/start. Recomputed from typedText itself
// rather than tracked separately, so it can never drift out of sync with
// what's actually been typed (including after backspacing mid-word).
function currentWordPrefix() {
  const m = typedText.match(/[A-Za-z']*$/);
  return m ? m[0] : "";
}

function matchCase(word, prefix) {
  if (prefix && prefix[0] !== prefix[0].toLowerCase()) {
    return word[0].toUpperCase() + word.slice(1);
  }
  return word;
}

function getSuggestions(prefix, n) {
  if (!prefix) return COMMON_WORDS.slice(0, n);
  const lower = prefix.toLowerCase();
  const seen = new Set();
  const matches = [];
  for (const w of COMMON_WORDS) {
    // COMMON_WORDS has a handful of incidental duplicates (hand-curated
    // across a few overlapping topic blocks) - skip repeats so the same
    // word can't take two of the five suggestion slots.
    if (w.startsWith(lower) && !seen.has(w)) {
      seen.add(w);
      matches.push(matchCase(w, prefix));
      if (matches.length >= n) break;
    }
  }
  return matches;
}

function updateSuggestions() {
  currentSuggestions = getSuggestions(currentWordPrefix(), 5);
  for (const cell of CELLS) {
    if (cell.kind === "suggestion") {
      cell.el.textContent = currentSuggestions[cell.idx] || "";
    }
  }
}

// Selecting a suggestion replaces the in-progress word (both in the
// local preview and, via the same backspaceKey()/typeChar() calls
// confirmKey() already uses for every other key, in whatever real
// window has OS focus) with the full word plus a trailing space.
function selectSuggestion(word) {
  const prefixLen = currentWordPrefix().length;
  for (let i = 0; i < prefixLen; i++) backspaceKey();
  for (const ch of word + " ") typeChar(ch);
}

function moveCursor(dir) {
  if (dir === "up") curRow = (curRow - 1 + GRID_ROWS) % GRID_ROWS;
  else if (dir === "down") curRow = (curRow + 1) % GRID_ROWS;
  else if (dir === "left") curCol = (curCol - 1 + GRID_COLS) % GRID_COLS;
  else if (dir === "right") curCol = (curCol + 1) % GRID_COLS;
  renderCursor();
}

function typeChar(ch) {
  typedText += ch;
  renderOutput();
  send({ type: "kb_type", char: ch });
}

function confirmKey() {
  const cell = currentCell();

  switch (cell.kind) {
    case "char":
      typeChar(capsOn ? cell.primary.toUpperCase() : cell.primary);
      break;
    case "shiftchar":
      typeChar(capsOn ? cell.shifted : cell.primary);
      break;
    case "space":
      typeChar(" ");
      break;
    case "enter":
      typedText += "\n";
      renderOutput();
      send({ type: "kb_enter" });
      break;
    case "backspace":
      typedText = typedText.slice(0, -1);
      renderOutput();
      send({ type: "kb_backspace" });
      break;
    case "tab":
      send({ type: "kb_special", key: "tab" });
      break;
    case "esc":
      send({ type: "kb_special", key: "esc" });
      break;
    case "caps":
      capsOn = !capsOn;
      cell.el.classList.toggle("is-active", capsOn);
      refreshLabels();
      break;
    case "volume_up":
      send({ type: "kb_special", key: "volume_up" });
      break;
    case "volume_down":
      send({ type: "kb_special", key: "volume_down" });
      break;
    case "brightness_up":
      send({ type: "kb_brightness", delta: 10 });
      break;
    case "brightness_down":
      send({ type: "kb_brightness", delta: -10 });
      break;
    case "mousetoggle":
      // The *only* way out of keyboard mode, deliberately not a
      // gesture - switches to eye/mouse mode, which hides this window.
      if (window.pywebview?.api?.enter_eye_mode) {
        window.pywebview.api.enter_eye_mode();
      } else {
        console.log("enter_eye_mode: no pywebview API available (dev mode)");
      }
      break;
    case "suggestion":
      if (currentSuggestions[cell.idx]) selectSuggestion(currentSuggestions[cell.idx]);
      break;
    case "poweronoff":
    case "blank":
      break; // deliberately no-op for now
  }

  cell.el.classList.add("is-confirmed");
  setTimeout(() => cell.el.classList.remove("is-confirmed"), 150);
}

function backspaceKey() {
  typedText = typedText.slice(0, -1);
  renderOutput();
  send({ type: "kb_backspace" });
}

// Called externally (from Python, via evaluate_js) whenever a gesture
// fires in the main window. Also used internally for physical-key
// testing below.
function applyGestureToKeyboard(label) {
  if (label === "up" || label === "down" || label === "left" || label === "right") {
    moveCursor(label);
  } else if (label === "confirm") {
    confirmKey();
  } else if (label === "backspace") {
    backspaceKey();
  }
}
window.applyGestureToKeyboard = applyGestureToKeyboard;

document.addEventListener("keydown", (e) => {
  const dirs = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right" };
  if (dirs[e.key]) {
    e.preventDefault();
    moveCursor(dirs[e.key]);
  } else if (e.key === "Enter") {
    e.preventDefault();
    confirmKey();
  } else if (e.key === "Backspace") {
    e.preventDefault();
    backspaceKey();
  }
});

buildKeyboardGrid();
renderOutput();
connectWS();
