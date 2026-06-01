"""파일 단위 Contextual Retrieval 용 AST skeleton 추출 + 문서 빌더.

ContextualEnricher 가 코드 청크의 맥락 문장을 만들 때 쓰는 "문서(document)" 는
그 청크가 나온 **파일 전체** 다. 큰 파일을 통째로 LLM 프롬프트에 넣으면 서버
n_ctx(4096 토큰) 를 넘기므로, 함수/메서드 **본문은 제거**하고 import 구문 +
최상위/클래스 레벨 정의 시그니처 + 바로 뒤 docstring 만 남긴 skeleton 으로
압축해서 문서로 쓴다.

실패 격리(불변 규칙): 이 모듈의 공개 함수(`extract_skeleton`,
`build_file_document`) 는 **어떤 입력에도 예외를 밖으로 던지지 않는다**. 파싱
실패/미지원 언어는 `extract_skeleton` 이 None 을 반환하고, `build_file_document`
은 raw 를 head 기준으로 truncate 한 최후 fallback 으로 떨어진다. 인덱싱 흐름이
이 모듈 때문에 깨지면 안 된다.

언어 확장 지점: 현재 **Python 만 구현**한다(RAGent 자체가 Python → 도그푸딩).
새 언어를 지원하려면
  1) `_LANG_BY_EXT` 에 확장자 → 언어 이름 매핑을 추가하고,
  2) `_SKELETON_EXTRACTORS` 에 언어 이름 → `(raw_code) -> str | None` 추출기를
     등록한다.
등록되지 않은 언어는 `extract_skeleton` 이 None 을 반환해 상위 fallback 을 탄다
(추측 매핑 금지 — astchunk 의 [Write] 함정과 동일하게 미구현은 명시적으로 None).
"""

import logging
import os

from ragent import config

logger = logging.getLogger("ragent")

# 제거된 함수/메서드 본문 자리에 남기는 마커.
_BODY_MARKER = "..."

# 한 단계 들여쓰기.
_INDENT = "    "

# 클래스 레벨 단순 statement(속성 할당 등) 를 skeleton 에 보존할 때의 길이 상한.
# 큰 데이터 리터럴을 통째로 끌고 들어오지 않도록 단일 라인 + 이 길이 이하만 남긴다.
_MAX_SIMPLE_STMT_CHARS = 200

# 확장자 → 언어 이름. astchunk 가 지원·설치하는 문법 바이너리와 같은 언어군.
_LANG_BY_EXT = {
    ".py": "python",
}


def _language_for(file_path: str) -> str | None:
    """파일 경로 확장자로 skeleton 추출기 언어 이름을 해석한다. 미지원 → None."""
    _, ext = os.path.splitext(file_path or "")
    return _LANG_BY_EXT.get(ext.lower())


# --------------------------------------------------------------------------- #
#  Python skeleton 추출 (tree-sitter)
# --------------------------------------------------------------------------- #
#  astchunk 가 이미 설치하는 tree-sitter 문법 바이너리(tree_sitter_python) 를
#  재사용한다. 파서 생성 패턴은 astchunk.astchunk_builder 의
#  `ts.Parser(ts.Language(tspython.language()))` 와 동일하다.
# --------------------------------------------------------------------------- #


def _node_text(node, src: bytes) -> str:
    """노드가 가리키는 원본 바이트 구간을 문자열로 디코딩한다."""
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _docstring_lines(body_node, src: bytes, indent: str) -> list[str]:
    """블록(또는 모듈) 의 첫 statement 가 docstring 이면 그 줄들을 반환한다.

    body_node 의 첫 named child 가 string 하나만 담은 expression_statement 일
    때만 docstring 으로 인정한다. indent 는 첫 줄에만 붙인다(이어지는 줄은 원본
    소스의 들여쓰기를 그대로 유지 — 소스와 skeleton 의 들여쓰기 단계가 같다).
    """
    if body_node is None:
        return []
    named = body_node.named_children
    if not named:
        return []
    first = named[0]
    if first.type != "expression_statement" or not first.named_children:
        return []
    string_node = first.named_children[0]
    if string_node.type != "string":
        return []
    return [indent + _node_text(string_node, src)]


def _render_definition(node, src: bytes, indent: str) -> list[str]:
    """function/class/decorated 정의 1개를 시그니처 + docstring 으로 렌더한다.

    함수/메서드 → 시그니처 + (docstring) + 본문 마커.
    클래스       → 헤더 + (docstring) + 멤버(메서드/중첩클래스/짧은 속성) 재귀,
                   멤버가 하나도 없으면 본문 마커.
    """
    if node.type == "decorated_definition":
        out: list[str] = []
        for child in node.children:
            if child.type == "decorator":
                out.append(indent + _node_text(child, src).strip())
        inner = node.child_by_field_name("definition")
        if inner is not None:
            out.extend(_render_definition(inner, src, indent))
        return out

    body = node.child_by_field_name("body")
    if body is None:
        # 본문이 없는(파싱상 비정상) 정의는 통째로 한 줄로.
        return [indent + _node_text(node, src).strip()]

    # 시그니처 = 노드 시작 ~ 본문 시작 직전 ("def f(...) -> T:" / "class C(Base):").
    sig = src[node.start_byte:body.start_byte].decode("utf-8", errors="replace").rstrip()
    out = [indent + sig]
    child_indent = indent + _INDENT

    doc = _docstring_lines(body, src, child_indent)
    out.extend(doc)

    if node.type == "class_definition":
        members: list[str] = []
        start = 1 if doc else 0  # docstring 은 이미 출력했으므로 건너뜀
        for child in body.named_children[start:]:
            if child.type in ("function_definition", "decorated_definition", "class_definition"):
                members.extend(_render_definition(child, src, child_indent))
            elif child.type == "expression_statement":
                # 클래스 레벨 속성 할당 등 — 짧은 단일 라인만 보존.
                text = _node_text(child, src)
                if "\n" not in text and len(text) <= _MAX_SIMPLE_STMT_CHARS:
                    members.append(child_indent + text.strip())
        if members:
            out.extend(members)
        else:
            out.append(child_indent + _BODY_MARKER)
    else:
        # 함수/메서드: 본문 제거 마커.
        out.append(child_indent + _BODY_MARKER)

    return out


def _render_module(root, src: bytes) -> list[str]:
    """모듈 루트의 최상위 노드를 순회해 skeleton 줄들을 만든다."""
    out: list[str] = []
    doc = _docstring_lines(root, src, "")
    out.extend(doc)
    start = 1 if doc else 0
    for child in root.named_children[start:]:
        t = child.type
        if t in ("import_statement", "import_from_statement", "future_import_statement"):
            out.append(_node_text(child, src).strip())
        elif t in ("function_definition", "decorated_definition", "class_definition"):
            out.extend(_render_definition(child, src, ""))
        elif t == "expression_statement":
            # 모듈 레벨 상수/속성 — 짧은 단일 라인만 보존(큰 데이터 리터럴 제외).
            text = _node_text(child, src)
            if "\n" not in text and len(text) <= _MAX_SIMPLE_STMT_CHARS:
                out.append(text.strip())
        # if __name__ == "__main__": 등 그 외 최상위 statement 는 skeleton 에서 제외.
    return out


def _extract_python_skeleton(raw_code: str) -> str | None:
    """Python 소스에서 import + 정의 시그니처 + docstring skeleton 을 추출한다.

    파싱/import 실패 시 None 을 반환하며 예외를 밖으로 던지지 않는다.
    """
    try:
        import tree_sitter as ts
        import tree_sitter_python as tspython
    except Exception:
        logger.exception("code_skeleton: tree-sitter python import unavailable")
        return None

    parser = ts.Parser(ts.Language(tspython.language()))
    src = raw_code.encode("utf-8")
    tree = parser.parse(src)
    skeleton = "\n".join(_render_module(tree.root_node, src)).strip()
    return skeleton or None


# 언어 이름 → 추출기. 새 언어는 여기에 등록한다.
_SKELETON_EXTRACTORS = {
    "python": _extract_python_skeleton,
}


# --------------------------------------------------------------------------- #
#  공개 API
# --------------------------------------------------------------------------- #


def extract_skeleton(raw_code: str, file_path: str) -> str | None:
    """파일 원문에서 AST skeleton 을 추출한다.

    지원 언어(`_LANG_BY_EXT`/`_SKELETON_EXTRACTORS`) 가 아니거나 파싱에 실패하면
    None 을 반환한다. 어떤 입력에도 예외를 밖으로 던지지 않는다.
    """
    if not raw_code:
        return None
    lang = _language_for(file_path)
    extractor = _SKELETON_EXTRACTORS.get(lang)
    if extractor is None:
        return None
    try:
        return extractor(raw_code)
    except Exception:
        logger.exception("code_skeleton: skeleton extraction failed for %s", file_path)
        return None


def build_file_document(raw_code: str, file_path: str) -> str:
    """코드 청크의 enrichment 문서로 쓸 파일 표현을 만든다.

    우선순위:
    1. 원문 길이 <= MAX_FILE_DOC_CHARS → 원문 그대로.
    2. 초과 → AST skeleton 시도. skeleton 이 있으면 그것을 문서로 쓰고, skeleton
       조차 상한을 넘으면 head 기준으로 자른다.
    3. skeleton 이 None(미지원/파싱 실패) → 원문을 head 기준으로 truncate 한
       최후 fallback.

    파일당 문서 모드(raw/skeleton/skeleton+truncated/truncated) 를 로그로 남긴다.
    예외를 밖으로 던지지 않는다.
    """
    raw_code = raw_code or ""
    cap = config.MAX_FILE_DOC_CHARS

    if len(raw_code) <= cap:
        logger.info(
            "code_skeleton: doc mode=raw path=%s chars=%d", file_path, len(raw_code)
        )
        return raw_code

    skeleton = extract_skeleton(raw_code, file_path)
    if skeleton:
        if len(skeleton) > cap:
            logger.info(
                "code_skeleton: doc mode=skeleton+truncated path=%s raw=%d skeleton=%d",
                file_path, len(raw_code), len(skeleton),
            )
            return skeleton[:cap].rstrip() + "\n[...truncated...]"
        logger.info(
            "code_skeleton: doc mode=skeleton path=%s raw=%d skeleton=%d",
            file_path, len(raw_code), len(skeleton),
        )
        return skeleton

    logger.info(
        "code_skeleton: doc mode=truncated path=%s raw=%d", file_path, len(raw_code)
    )
    return raw_code[:cap].rstrip() + "\n[...truncated...]"
