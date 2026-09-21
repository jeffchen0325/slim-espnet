#!/bin/bash
set -euo pipefail


# 基于脚本自身位置定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# ==========================================
# 【配置区】依赖列表
# 格式: "PKG_NAME|PKG_VERSION|TAG|REPO_URL"
# ==========================================
DEPENDENCIES=(
    "espnet|202609|v.202609|https://github.com/espnet/espnet"
    "espnet_model_zoo|202111|v0.1.7|https://github.com/espnet/espnet_model_zoo"
)

# ==========================================
# 核心归档函数
# ==========================================
archive_license() {
    local PKG_NAME="$1"
    local PKG_VERSION="$2"
    local TAG="$3"
    local REPO_URL="$4"

    echo "🔄 Processing ${PKG_NAME}-${PKG_VERSION} ..."

    local TMP_DIR="/tmp/${PKG_NAME}-${PKG_VERSION}"
    rm -rf "$TMP_DIR"

    # 1. Clone
    if ! timeout 120 git clone --depth 1 --branch "$TAG" "$REPO_URL" "$TMP_DIR" 2>/dev/null; then
        echo "❌ ERROR: Failed to clone ${PKG_NAME} (tag: ${TAG}). Skipping." >&2
        return 1
    fi

    # 2. Extract SHA
    local COMMIT_SHA
    COMMIT_SHA=$(git -C "$TMP_DIR" rev-parse HEAD)
    if [ -z "$COMMIT_SHA" ]; then
        echo "❌ ERROR: Failed to get commit SHA for ${PKG_NAME}. Skipping." >&2
        rm -rf "$TMP_DIR"
        return 1
    fi

    local ARCHIVE_DIR="${SCRIPT_DIR}/third-party-licenses/${PKG_NAME}-${PKG_VERSION}"
    mkdir -p "$ARCHIVE_DIR"

    # 3. Copy LICENSE
    if [ ! -f "$TMP_DIR/LICENSE" ]; then
        echo "⚠️  WARNING: LICENSE not found for ${PKG_NAME}" >&2
        echo "LICENSE_NOT_FOUND" > "$ARCHIVE_DIR/LICENSE.status"
    else
        cp "$TMP_DIR/LICENSE" "$ARCHIVE_DIR/LICENSE"
    fi

    # 4. Write provenance
    cat > "$ARCHIVE_DIR/provenance.txt" <<EOF
Package: $PKG_NAME
Version: $PKG_VERSION
Source Repo: $REPO_URL
Tag: $TAG
Commit SHA: $COMMIT_SHA
Retrieved At: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Method: git clone --depth 1 --branch $TAG
EOF

    # 5. Cleanup
    rm -rf "$TMP_DIR"
    echo "✅ Archived: ${ARCHIVE_DIR}"
}

# ==========================================
# 自定义许可证分发逻辑（已精简）
# ==========================================
distribute_license() {
    local PKG_NAME="$1"
    local ARCHIVE_DIR="$2"
    local LICENSE_FILE="${ARCHIVE_DIR}/LICENSE"

    [ -f "$LICENSE_FILE" ] || return 0

    case "$PKG_NAME" in
        espnet)
            local NOTICE_FILE="${PROJECT_ROOT}/NOTICE"
            echo "📝 Processing ${PKG_NAME} attribution..."

            if [ ! -f "$NOTICE_FILE" ]; then
                echo "ESPnet3 Third-Party Notices" > "$NOTICE_FILE"
                echo "=============================" >> "$NOTICE_FILE"
            fi

            # 衍生声明（精确匹配 + 安全退出码）
            has_attribution=false
            grep -qF "This product is based on software developed at:" "$NOTICE_FILE" 2>/dev/null && has_attribution=true
            if [ "$has_attribution" = false ]; then
                {
                    echo ""
                    echo "---"
                    echo "This product is based on software developed at:"
                    echo "- ${PKG_NAME} (https://github.com/espnet/espnet)"
                    echo "  Original License: See compliance/third-party-licenses/${PKG_NAME}-*/LICENSE"
                } >> "$NOTICE_FILE"
            else
                echo "   ⏭️  Attribution already exists, skipping."
            fi

            # 机械性变更声明（heredoc 无缩进污染 + 安全退出码）
            read -r -d '' MECHANICAL_NOTE <<'EOF' || true
A significant number of source files have been renamed, relocated, or had their import paths updated as part of the ESPnet3 restructuring. Where such changes are purely mechanical and do not alter the functional logic or creative expression of the original code, the original copyright status is preserved unchanged. These files remain attributed to the original ESPnet authors via this NOTICE and the archived licenses in compliance/third-party-licenses/.
EOF

            has_mechanical=false
            grep -qF "purely mechanical and do not alter" "$NOTICE_FILE" 2>/dev/null && has_mechanical=true
            if [ "$has_mechanical" = false ]; then
                {
                    echo ""
                    echo "---"
                    echo "Mechanical Changes Notice:"
                    echo "$MECHANICAL_NOTE"
                } >> "$NOTICE_FILE"
                echo "📝 Appended mechanical-changes notice."
            else
                echo "   ⏭️  Mechanical-changes notice already exists, skipping."
            fi
            ;;

        espnet_model_zoo)
            # 微量修改：作为子目录唯一 LICENSE
            local TARGET_DIR="${PROJECT_ROOT}/espnet3/model_zoo"
            mkdir -p "$TARGET_DIR"
            cp -f "$LICENSE_FILE" "${TARGET_DIR}/LICENSE"
            echo "📄 Placed ${PKG_NAME} LICENSE as sole license for model_zoo/"
            ;;

        *)
            # 其他依赖仅归档
            ;;
    esac
}

# ==========================================
# 主执行流程
# ==========================================
SUCCESS=0
FAIL=0

if [ ${#DEPENDENCIES[@]} -eq 0 ]; then
    echo "⚠️ No dependencies defined."
    exit 0
fi

for entry in "${DEPENDENCIES[@]}"; do
    IFS='|' read -r name version tag url <<< "$entry"
    if archive_license "$name" "$version" "$tag" "$url"; then
        SUCCESS=$((SUCCESS + 1))
        distribute_license "$name" "${SCRIPT_DIR}/third-party-licenses/${name}-${version}"
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "📊 Summary: ${SUCCESS} succeeded, ${FAIL} failed (Total: ${#DEPENDENCIES[@]})"
echo "=========================================="

[ "$FAIL" -eq 0 ] || exit 1
