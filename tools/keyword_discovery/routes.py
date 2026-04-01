"""Flask blueprint for the Keyword Discovery tool."""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from flask import (
    Blueprint,
    render_template,
    request,
    session,
    jsonify,
    send_file,
)

from .corpus import Corpus, load_corpus
from .discovery import (
    CooccurrenceDiscovery,
    SemanticDiscovery,
    NetworkDiscovery,
    CorpusSpecificDiscovery,
)
from .discovery.network import export_graphml
from .models import Candidate, ChainRound, ChainState

logger = logging.getLogger(__name__)

keyword_discovery_bp = Blueprint(
    "keyword_discovery", __name__, template_folder="templates"
)

# ---------------------------------------------------------------------------
# Server-side caches (keyed by session id)
# ---------------------------------------------------------------------------
_corpus_cache: dict[str, Corpus] = {}
_state_cache: dict[str, ChainState] = {}
_corpus_lock = Lock()
_MAX_CACHE = 20

# Network module instance kept per-session so we can export the graph
_network_modules: dict[str, NetworkDiscovery] = {}

# Comprehensive English stopwords — covers function words, common verbs/adverbs,
# and generic terms that appear in any topic without carrying domain signal.
_STOPWORDS = frozenset((
    # determiners, pronouns, prepositions, conjunctions
    "a an the and or but in on at to for of is it this that was were be been "
    "being have has had do does did will would shall should may might can could "
    "i me my we our you your he she they them his her its not no with from by "
    "as are am if so than too very just about also how all each which their "
    "there here when where what who whom why into then some any more most "
    "other such only over after before between through during these those "
    "out off down few both same own even still already while since because "
    "another many much us him get got "
    # common verbs
    "like use used make made go going take come went see said know think say "
    "look want give let tell work try need feel become leave call find "
    "keep put set run show help turn move live play write read start "
    "hold bring hear begin seem stand happen change follow lead open mean "
    "believe ask close include continue consider learn end remember love "
    "pay provide build meet add offer stop watch lose send sit "
    "create speak buy wait hope talk agree carry "
    # common nouns (too generic to be keywords)
    "people time right day thing way back point part place case hand "
    "fact group number problem question state side area line result "
    "moment form home word name school company house head end job body "
    "information family power country system program week water city "
    "story children world life year man woman money title "
    # common adjectives / adverbs
    "good new first last long great old big little small next early young "
    "important few large sure free right whole real better best bad able "
    "possible different sure enough far away together ago today least "
    "already something nothing everything anything often never always "
    "sometimes usually probably maybe perhaps really actually well quite "
    "almost enough ever yet still again soon later away around "
    "however rather therefore indeed certainly clearly simply "
    "especially actually basically certainly finally generally merely "
    "nearly perhaps probably actually forever forward "
    "above across along among behind below beneath beside beyond "
    "inside outside throughout upon whether without within "
    # discourse / filler
    "yes yeah no ok well oh hey hi hello please thanks thank sorry "
    "just gonna wanna gotta kinda sorta whatever actually literally "
    "basically totally exactly definitely absolutely clearly obviously "
    "according says said told asked given called known shown based "
    "using including looking making taking getting doing having seeing "
    "going coming saying trying working playing "
    # common social media / web noise
    "http https www com org net html pic link post share like comment "
    "video photo image view click read via rt per re vs e g etc "
    "s t d m ll ve re didn doesn isn wasn weren hasn hadn won wouldn "
    "shouldn couldn ain "
    # numerals and short tokens
    "one two three four five six seven eight nine ten hundred thousand "
    "million billion first second third last "
    # more generic terms
    "lot lots kind sort type example examples thing things stuff bit "
    "ones those much such gets put puts got getting goes gone done "
    "makes comes takes gives tells looks seems feels means keeps "
    "wants needs tries works helps starts shows plays runs moves sets "
    "dark light full half single double main true false entire whole "
    "total complete final major minor key top high low level "
    "real clear hard pretty certain simply happen happened happening "
    "plan process reason sense fact truth question answer response "
    "note report article source image content page site original "
    "entire amount rate increase large significant "
    "recent update share shared sharing post posted posting "
    "tens hundreds thousands millions billions "
    # generic verbs / adjectives that appear near any topic
    "exposed locked funded funding organizations organized "
    "worldwide globally apparently allegedly revealed leading "
    "involved massive huge growing bigger biggest supported "
    "claimed claims claiming pushed pushing spread spreading "
    "proven supposed working running created moved turned "
    "taken placed gone brought allowed forced held sent known seen "
    "written spoken covered listed expected given taken signed "
).split())

# Maximum document frequency as fraction of corpus — terms appearing in more
# than this fraction of documents are too common to be informative.
MAX_DF_RATIO = 0.04


def _get_corpus_id() -> str:
    """Return or create a corpus cache key for the current session."""
    cid = session.get("kd_corpus_id")
    if not cid:
        cid = uuid.uuid4().hex[:12]
        session["kd_corpus_id"] = cid
    return cid


def _get_chain_state() -> ChainState:
    """Retrieve chain state from server-side cache."""
    cid = session.get("kd_corpus_id")
    if cid and cid in _state_cache:
        return _state_cache[cid]
    return ChainState()


def _save_chain_state(state: ChainState) -> None:
    """Persist chain state to server-side cache."""
    cid = session.get("kd_corpus_id")
    if cid:
        _state_cache[cid] = state


def _get_corpus() -> Corpus | None:
    cid = session.get("kd_corpus_id")
    if cid and cid in _corpus_cache:
        return _corpus_cache[cid]
    return None


# ---------------------------------------------------------------------------
# Candidate merging
# ---------------------------------------------------------------------------

def _is_stem_overlap(term: str, seed_set: set[str]) -> bool:
    """Return True if *term* is a trivial morphological variant of any seed.

    Catches cases like seed "rothschild" producing candidate "rothschilds"
    or "the rothschilds".  Works both directions: a candidate whose words
    are all prefixes/extensions of a seed, or vice-versa.
    """
    term_words = term.split()
    for seed in seed_set:
        seed_words = seed.split()
        # Single-word seed vs every word in the candidate
        if len(seed_words) == 1:
            sw = seed_words[0]
            if any(w.startswith(sw) or sw.startswith(w) for w in term_words):
                return True
        # Multi-word: check if all seed words overlap with all candidate words
        elif len(seed_words) == len(term_words):
            if all(
                cw.startswith(sw) or sw.startswith(cw)
                for sw, cw in zip(seed_words, term_words)
            ):
                return True
    return False


def _merge_candidates(
    all_candidates: list[Candidate],
    seed_set: set[str],
    dismissed: set[str],
    excluded: set[str],
    min_df: int,
    corpus_size: int = 0,
) -> list[Candidate]:
    """Deduplicate, merge multi-source candidates, and filter."""
    # Combine seeds and excludes for stem-overlap checking
    stem_block = seed_set | excluded
    max_df = int(corpus_size * MAX_DF_RATIO) if corpus_size > 0 else float("inf")
    best: dict[str, Candidate] = {}

    for c in all_candidates:
        term = c.term.lower().strip()
        # Filter
        if term in seed_set or term in dismissed or term in excluded:
            continue
        if _is_stem_overlap(term, stem_block):
            continue
        if len(term) < 2:
            continue
        if term in _STOPWORDS:
            continue
        # Check any individual word in the term is a stopword (for bigrams like "the people")
        if all(w in _STOPWORDS for w in term.split()):
            continue
        if c.doc_count < min_df and c.source != "coded_language":
            continue
        if c.doc_count > max_df:
            continue

        if term not in best:
            best[term] = Candidate(
                term=term,
                score=c.score,
                source=c.source,
                sources=list(c.sources),
                context=c.context,
                evidence=list(c.evidence),
                community_id=c.community_id,
                doc_count=c.doc_count,
            )
        else:
            existing = best[term]
            # Take highest score
            if c.score > existing.score:
                existing.score = c.score
                existing.context = c.context
            # Merge sources
            for src in c.sources:
                if src not in existing.sources:
                    existing.sources.append(src)
            existing.source = "|".join(existing.sources)
            # Merge evidence
            merged = list(dict.fromkeys(existing.evidence + c.evidence))[:5]
            existing.evidence = merged
            # Keep community_id if set
            if c.community_id is not None:
                existing.community_id = c.community_id

    result = sorted(best.values(), key=lambda c: c.score, reverse=True)

    # Subsumption: if a unigram appears as a word inside a higher-scoring
    # bigram candidate, drop the unigram (e.g. "deep" subsumed by "deep state")
    bigram_words: set[str] = set()
    for c in result:
        words = c.term.split()
        if len(words) >= 2:
            for w in words:
                bigram_words.add(w)
    result = [c for c in result if " " in c.term or c.term not in bigram_words]

    return result


# ---------------------------------------------------------------------------
# Reranker (VoyageAI)
# ---------------------------------------------------------------------------

def _rerank_candidates(
    candidates: list[Candidate],
    seed_keywords: list[str],
    top_k: int | None = None,
) -> list[Candidate]:
    """Rerank candidates using VoyageAI's reranker.

    Each candidate is represented as its term + context + a sample of evidence.
    The query is the seed keyword list framed as a discovery task.
    Falls back to the original order if the API key is missing or the call fails.
    """
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    if not api_key or not candidates:
        return candidates

    try:
        import voyageai
    except ImportError:
        logger.warning("voyageai package not installed, skipping reranker")
        return candidates

    # Build query: the seed keywords framed as what we're looking for
    query = (
        f"Keywords and terms related to: {', '.join(seed_keywords)}. "
        "Looking for ideological signals, jargon, coded language, named figures, "
        "organisations, slogans, and domain-specific vocabulary."
    )

    # Build documents: each candidate as a compact text block
    documents: list[str] = []
    for c in candidates:
        parts = [f"Term: {c.term}"]
        if c.context:
            parts.append(f"Context: {c.context}")
        if c.evidence:
            parts.append(f"Example: {c.evidence[0][:200]}")
        documents.append(" | ".join(parts))

    try:
        vo = voyageai.Client(api_key=api_key)
        result = vo.rerank(
            query=query,
            documents=documents,
            model="rerank-2.5",
            top_k=top_k or len(candidates),
        )

        # Use VoyageAI's relevance score directly
        reranked: list[Candidate] = []
        for r in result.results:
            c = candidates[r.index]
            c.score = round(r.relevance_score, 4)
            reranked.append(c)
        logger.info("Reranked %d candidates via VoyageAI", len(reranked))
        return reranked

    except Exception:
        logger.exception("VoyageAI reranking failed, returning original order")
        return candidates


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@keyword_discovery_bp.route("/", methods=["GET"])
def index():
    """Render the main keyword discovery page."""
    state = _get_chain_state()
    corpus = _get_corpus()
    has_corpus = corpus is not None
    return render_template(
        "keyword_discovery.html",
        state=state,
        has_corpus=has_corpus,
    )


@keyword_discovery_bp.route("/upload", methods=["POST"])
def upload():
    """Upload and parse a corpus file."""
    file = request.files.get("corpus_file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    user_column = request.form.get("text_column", "").strip() or None
    ngram_str = request.form.get("ngram_range", "2")
    ngram_max = max(1, min(3, int(ngram_str))) if ngram_str.isdigit() else 2
    min_df = max(1, int(request.form.get("min_df", "2")))

    file_bytes = file.read()
    try:
        corpus = load_corpus(
            file_bytes,
            file.filename,
            user_column=user_column,
            ngram_range=(1, ngram_max),
            min_df=min_df,
        )
    except Exception as e:
        logger.exception("Corpus load failed")
        return jsonify({"error": f"Failed to load corpus: {e}"}), 400

    # Cache corpus
    cid = _get_corpus_id()
    with _corpus_lock:
        # Evict oldest if cache is full
        while len(_corpus_cache) >= _MAX_CACHE:
            oldest = next(iter(_corpus_cache))
            del _corpus_cache[oldest]
            _state_cache.pop(oldest, None)
        _corpus_cache[cid] = corpus

    # Reset chain state
    state = ChainState(
        corpus_filename=file.filename,
        text_column=corpus.text_column,
        total_docs=len(corpus.df),
        unique_terms=len(corpus.vocab),
    )
    _save_chain_state(state)

    return jsonify({
        "ok": True,
        "filename": file.filename,
        "text_column": corpus.text_column,
        "total_docs": len(corpus.df),
        "unique_terms": len(corpus.vocab),
    })


@keyword_discovery_bp.route("/discover", methods=["POST"])
def discover():
    """Run all discovery modules and return merged candidates as JSON."""
    corpus = _get_corpus()
    if corpus is None:
        return jsonify({"error": "No corpus loaded. Upload a file first."}), 400

    data = request.get_json(silent=True) or {}
    seeds_raw = data.get("seeds", "")
    exclude_raw = data.get("exclude", "")
    min_df = max(1, int(data.get("min_df", 5)))
    top_n = max(1, min(200, int(data.get("top_n", 50))))

    # Parse seeds
    seeds = [s.strip() for s in seeds_raw.split(",") if s.strip()]
    if not seeds:
        return jsonify({"error": "No seed keywords provided."}), 400

    # Parse exclude list — merge form input with accumulated state
    excluded_form = {s.strip().lower() for s in exclude_raw.split(",") if s.strip()}

    seed_set = {s.lower() for s in seeds}
    state = _get_chain_state()
    dismissed_set = set(state.dismissed)
    excluded = excluded_form | set(state.excluded)

    # Run discovery modules in parallel
    cooc = CooccurrenceDiscovery()
    sem = SemanticDiscovery()
    net = NetworkDiscovery()
    csp = CorpusSpecificDiscovery()

    cid = session.get("kd_corpus_id", "")
    _network_modules[cid] = net

    all_candidates: list[Candidate] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(mod.discover, seeds, corpus, top_n, min_df): name
            for name, mod in [
                ("cooccurrence", cooc),
                ("semantic", sem),
                ("network", net),
                ("corpus_specific", csp),
            ]
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                all_candidates.extend(result)
            except Exception as e:
                logger.exception("Discovery module %s failed", name)
                errors.append(f"{name}: {e}")

    merged = _merge_candidates(all_candidates, seed_set, dismissed_set, excluded, min_df,
                                corpus_size=len(corpus.clean_texts))

    # Rerank via VoyageAI if available
    merged = _rerank_candidates(merged, seeds, top_k=top_n)

    # Update chain state with current seeds
    state.current_seeds = seeds

    # Store candidates in the latest round (or create a new round)
    round_number = len(state.rounds) + 1
    new_round = ChainRound(
        round_number=round_number,
        seed_keywords=list(seeds),
        candidates=[c.model_copy() for c in merged],
    )
    state.rounds.append(new_round)
    _save_chain_state(state)

    return jsonify({
        "ok": True,
        "round_number": round_number,
        "seed_count": len(seeds),
        "candidate_count": len(merged),
        "candidates": [c.model_dump() for c in merged],
        "errors": errors,
    })


@keyword_discovery_bp.route("/select", methods=["POST"])
def select():
    """Add selected terms to seeds, exclude list, or dismiss."""
    data = request.get_json(silent=True) or {}
    added = [t.strip().lower() for t in data.get("added", []) if t.strip()]
    excluded_new = [t.strip().lower() for t in data.get("excluded", []) if t.strip()]
    dismissed = [t.strip().lower() for t in data.get("dismissed", []) if t.strip()]

    state = _get_chain_state()

    # Update the current round
    if state.rounds:
        current_round = state.rounds[-1]
        current_round.added_terms = added
        current_round.excluded_terms = excluded_new
        current_round.dismissed_terms = dismissed

    # Extend seeds
    current_seeds_set = {s.lower() for s in state.current_seeds}
    for term in added:
        if term not in current_seeds_set:
            state.current_seeds.append(term)
            current_seeds_set.add(term)

    # Extend exclude list
    excluded_set = set(state.excluded)
    for term in excluded_new:
        if term not in excluded_set:
            state.excluded.append(term)
            excluded_set.add(term)

    # Extend dismissed list
    for term in dismissed:
        if term not in state.dismissed:
            state.dismissed.append(term)

    _save_chain_state(state)

    return jsonify({
        "ok": True,
        "current_seeds": state.current_seeds,
        "excluded": state.excluded,
        "dismissed_count": len(state.dismissed),
    })


@keyword_discovery_bp.route("/undo", methods=["POST"])
def undo():
    """Revert to the previous chain round."""
    state = _get_chain_state()

    if not state.rounds:
        return jsonify({"error": "No rounds to undo."}), 400

    removed_round = state.rounds.pop()

    # Remove added terms from current seeds
    added_set = set(removed_round.added_terms)
    state.current_seeds = [s for s in state.current_seeds if s.lower() not in added_set]

    # Remove excluded terms
    excluded_set = set(removed_round.excluded_terms)
    state.excluded = [e for e in state.excluded if e not in excluded_set]

    # Remove dismissed terms
    dismissed_set = set(removed_round.dismissed_terms)
    state.dismissed = [d for d in state.dismissed if d not in dismissed_set]

    _save_chain_state(state)

    return jsonify({
        "ok": True,
        "current_seeds": state.current_seeds,
        "rounds_remaining": len(state.rounds),
    })


@keyword_discovery_bp.route("/export", methods=["GET"])
def export():
    """Download results in various formats."""
    fmt = request.args.get("format", "json")
    state = _get_chain_state()

    if fmt == "json":
        payload = state.model_dump_json(indent=2)
        buf = io.BytesIO(payload.encode("utf-8"))
        return send_file(buf, mimetype="application/json", as_attachment=True,
                         download_name="keyword_discovery.json")

    if fmt == "csv":
        import csv as csv_mod

        buf = io.StringIO()
        writer = csv_mod.writer(buf)
        writer.writerow(["term", "score", "sources", "context", "doc_count", "community_id", "round"])
        for rnd in state.rounds:
            for c in rnd.candidates:
                writer.writerow([
                    c.term, c.score, "|".join(c.sources), c.context,
                    c.doc_count, c.community_id or "", rnd.round_number,
                ])
        mem = io.BytesIO(buf.getvalue().encode("utf-8"))
        return send_file(mem, mimetype="text/csv", as_attachment=True,
                         download_name="keyword_discovery.csv")

    if fmt == "boolean":
        terms = state.current_seeds
        query = " OR ".join(f'"{t}"' if " " in t else t for t in terms)
        buf = io.BytesIO(query.encode("utf-8"))
        return send_file(buf, mimetype="text/plain", as_attachment=True,
                         download_name="boolean_query.txt")

    if fmt == "graphml":
        cid = session.get("kd_corpus_id", "")
        net_mod = _network_modules.get(cid)
        if net_mod and net_mod.last_graph:
            data = export_graphml(net_mod.last_graph)
            buf = io.BytesIO(data)
            return send_file(buf, mimetype="application/xml", as_attachment=True,
                             download_name="keyword_network.graphml")
        return jsonify({"error": "No network graph available. Run discovery with network module first."}), 400

    return jsonify({"error": f"Unknown format: {fmt}"}), 400
