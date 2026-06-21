"""
'전황' 폴더 전체 글에서 주식 투자 관련 핵심 내용을 사내 LLM(env.txt 설정)으로
카테고리별 원문 발췌(인용) 추출하여 하나의 결과 파일로 합치는 스크립트.

사용법 (가상환경 python 사용):
    .venv\\Scripts\\python.exe 전황_result\\extract.py
    .venv\\Scripts\\python.exe 전황_result\\extract.py --force   # 캐시 무시하고 재추출
    .venv\\Scripts\\python.exe 전황_result\\extract.py --limit 5  # 테스트용으로 5개 파일만

env.txt (저장소 루트)에서 OPENAI_API_KEY / LLM_MODEL / LLM_BASE_URL을 읽어 사용한다.
"""

import argparse
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent          # D:\2026_Agent\3_주시기
SOURCE_DIR = ROOT_DIR / "전황"
ENV_PATH = ROOT_DIR / "env.txt"
CACHE_DIR = SCRIPT_DIR / "_cache"
OUTPUT_PATH = SCRIPT_DIR / "전황_투자정보_추출.md"

# 결과에 포함할 카테고리 (순서 고정)
CATEGORY_KEYS = ["시황", "종목 선정", "마인드", "매수", "매도", "손절매", "기타"]
CATEGORY_LABELS = {
    "시황": "시황",
    "종목 선정": "종목 선정",
    "마인드": "마인드",
    "매수": "매수",
    "매도": "매도",
    "손절매": "손절매",
    "기타": "기타 투자 인사이트",
}

# (표시용 폴더명, 실제 디렉토리, recursive 여부) - 이 순서로 최종 결과에 배치
SECTIONS = [
    ("전황", SOURCE_DIR, False),
    ("전황_카페", SOURCE_DIR / "전황_카페", False),
    ("chamberine3/전황의 주식철학", SOURCE_DIR / "chamberine3" / "전황의 주식철학", False),
    ("chamberine3/전황트레이딩뷰", SOURCE_DIR / "chamberine3" / "전황트레이딩뷰", False),
    ("chamberine3/정보 나눔의 공간", SOURCE_DIR / "chamberine3" / "정보 나눔의 공간", False),
]

SYSTEM_PROMPT = """당신은 한국 주식 트레이더의 블로그/카페 글을 분석하는 투자 리서치 어시스턴트입니다.
주어진 글에서 아래 6개 카테고리에 해당하는 "주식 투자 관련 핵심 문장/문단"을 찾아 JSON으로 추출하세요.

카테고리:
- 시황: 시장 전반에 대한 분석, 추세, 지수, 매크로 등에 대한 코멘트
- 종목 선정: 특정 종목/섹터를 고르는 기준, 관심종목, 종목 분석
- 마인드: 매매 심리, 마음가짐, 태도, 멘탈 관리에 대한 통찰
- 매수: 매수 시점, 매수 기준, 진입 전략
- 매도: 매도 시점, 매도 기준, 차익실현 전략
- 손절매: 손절 기준, 리스크 관리, 손실 대응
- 기타: 위 6개에는 안 맞지만 투자에 중요한 인사이트 (위 항목에 들어가지 않는 것만, 무리하게 채우지 말 것)

매우 중요한 규칙:
1. 절대 의역하거나 요약하지 말고, 원문에 있는 문장/문단을 "그대로" 복사해서 인용하세요 (verbatim). 단어를 바꾸거나 다듬지 마세요.
2. 각 인용은 원문에 실제로 존재하는 연속된 텍스트여야 합니다.
3. 해당 카테고리에 맞는 내용이 글에 없으면 빈 배열 []을 반환하세요. 억지로 채우지 마세요.
4. 단순 공지/일정 안내/인사말처럼 투자 내용이 전혀 없는 글이면 모든 카테고리를 빈 배열로 반환하세요.
5. 반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{"시황": ["...", "..."], "종목 선정": ["..."], "마인드": ["..."], "매수": ["..."], "매도": ["..."], "손절매": ["..."], "기타": ["..."]}
"""


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s)


def list_txt_files(directory: Path) -> list:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".txt" and p.is_file())


def cache_path_for(relpath: str) -> Path:
    safe = re.sub(r"[^\w\-.]+", "_", relpath)
    return CACHE_DIR / f"{safe}.json"


def call_llm(client: OpenAI, model: str, text: str) -> dict:
    last_err = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=16000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"다음은 분석할 글 원문입니다:\n\n{text}"},
                ],
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            return {k: data.get(k, []) or [] for k in CATEGORY_KEYS}
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM 호출 실패: {last_err}")


def process_file(client: OpenAI, model: str, relpath: str, path: Path, force: bool) -> dict:
    cpath = cache_path_for(relpath)
    if not force and cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))["result"]

    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        result = {k: [] for k in CATEGORY_KEYS}
    else:
        raw = call_llm(client, model, text)
        result = {}
        dropped = 0
        for k in CATEGORY_KEYS:
            kept = []
            for quote in raw.get(k, []):
                if isinstance(quote, str) and normalize(quote) and normalize(quote) in normalize(text):
                    kept.append(quote.strip())
                else:
                    dropped += 1
            result[k] = kept
        if dropped:
            print(f"  [경고] {relpath}: 원문과 일치하지 않아 제외된 인용 {dropped}건")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath.write_text(
        json.dumps({"relpath": relpath, "result": result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_markdown(all_results: list) -> str:
    lines = ["# 전황 투자 정보 추출 결과", ""]
    lines.append("> 시황 / 종목 선정 / 마인드 / 매수 / 매도 / 손절매 관련 내용을 원문 그대로 발췌한 결과입니다.")
    lines.append("")
    included = 0
    for relpath, result in all_results:
        if all(not result.get(k) for k in CATEGORY_KEYS):
            continue
        included += 1
        lines.append(f"## 출처: {relpath}")
        lines.append("")
        for k in CATEGORY_KEYS:
            quotes = result.get(k, [])
            if not quotes:
                continue
            lines.append(f"### {CATEGORY_LABELS[k]}")
            for q in quotes:
                q_oneline = q.replace("\n", " ").strip()
                lines.append(f"- \"{q_oneline}\"")
            lines.append("")
        lines.append("")
    print(f"투자 관련 내용이 있는 파일: {included} / {len(all_results)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="캐시 무시하고 전체 재추출")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 처음 N개 파일만 처리")
    parser.add_argument("--workers", type=int, default=5, help="동시 LLM 호출 수")
    args = parser.parse_args()

    env = load_env(ENV_PATH)
    api_key = env.get("OPENAI_API_KEY")
    model = env.get("LLM_MODEL", "gpt-4o")
    base_url = env.get("LLM_BASE_URL") or None
    if not api_key:
        raise SystemExit(f"OPENAI_API_KEY를 {ENV_PATH} 에서 찾을 수 없습니다.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    targets = []  # (relpath, path)
    for label, directory, _ in SECTIONS:
        for f in list_txt_files(directory):
            relpath = f"{label}/{f.name}"
            targets.append((relpath, f))

    if args.limit:
        targets = targets[: args.limit]

    print(f"총 {len(targets)}개 파일 처리 시작 (model={model}, workers={args.workers})")

    results_map = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_file, client, model, relpath, path, args.force): relpath
            for relpath, path in targets
        }
        done = 0
        for fut in as_completed(futures):
            relpath = futures[fut]
            done += 1
            try:
                results_map[relpath] = fut.result()
            except Exception as e:
                print(f"  [실패] {relpath}: {e}")
                results_map[relpath] = {k: [] for k in CATEGORY_KEYS}
            print(f"[{done}/{len(targets)}] {relpath}")

    ordered_results = [(relpath, results_map[relpath]) for relpath, _ in targets]
    markdown = build_markdown(ordered_results)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"완료. 결과 파일: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
