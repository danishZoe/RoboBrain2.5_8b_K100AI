#!/usr/bin/env bash
# =============================================================================
# 当前物理卡指纹：用于「同卡 A/B」校验
#
# 背景：SCNet 容器被平台回收重建后**物理卡会更换**（2026-11-01 实测）。
#       跨卡的绝对值不可比（实测同配置跨卡可差约 30%），因此压测目录里
#       必须记录卡指纹，报告脚本据此判断两轮数据是否来自同一张卡。
#
# 用法：
#   source scripts/card_identity.sh     # 导出 CARD_ID / CARD_UID / CARD_NAME / CARD_GFX
#   bash scripts/card_identity.sh        # 只打印
#
# 变量：
#   CARD_UID      卡的唯一标识（GPU unique id / serial；取不到则为空）
#   CARD_NAME     型号名
#   CARD_GFX      架构名（如 gfx928）
#   CARD_ID       综合指纹：
#                   CARD_UID 非空 → "uid:<唯一标识>"
#                   否则          → "host:<主机名>:boot:<boot_id>"
#                 （后者能区分「换容器/换节点」，但**不能**区分同节点换卡，
#                   因此此时报告里的同卡结论必须标注「同卡未确认」）
#   CARD_ID_FILE  指纹落盘位置（默认 $WS/outputs/CARD_ID.txt）
#
# 永不失败：任何探测失败都只影响字段是否为空，不影响脚本退出码。
# =============================================================================
_ci_ws="${WS:-/root/private_data/workspace/robo}"
_ci_uid=""
_ci_name=""
_ci_gfx=""

# --- 1) rocm-smi（部分镜像没有该工具）-----------------------------------------
if command -v rocm-smi >/dev/null 2>&1; then
  _ci_smi="$(rocm-smi --showuniqueid --showproductname 2>/dev/null || true)"
  if [ -n "$_ci_smi" ]; then
    _ci_u="$(printf '%s\n' "$_ci_smi" | grep -i 'unique *id' | head -n 1 \
             | sed 's/.*: *//' | tr -d '\r' | sed 's/^ *//;s/ *$//')"
    [ -n "$_ci_u" ] && _ci_uid="$_ci_u"
    _ci_n="$(printf '%s\n' "$_ci_smi" | grep -iE 'card (series|model|sku)' | head -n 1 \
             | sed 's/.*: *//' | tr -d '\r' | sed 's/^ *//;s/ *$//')"
    [ -n "$_ci_n" ] && _ci_name="$_ci_n"
  fi
fi

# --- 2) sysfs unique_id / serial ----------------------------------------------
if [ -z "$_ci_uid" ]; then
  for _ci_f in /sys/class/drm/card0/device/unique_id /sys/class/drm/card0/device/serial; do
    if [ -r "$_ci_f" ]; then
      _ci_v="$(tr -d '\r\n' < "$_ci_f" 2>/dev/null || true)"
      _ci_v="$(printf '%s' "$_ci_v" | sed 's/^ *//;s/ *$//')"
      if [ -n "$_ci_v" ]; then _ci_uid="$_ci_v"; break; fi
    fi
  done
fi

# --- 3) torch（HIP 下 torch.cuda 映射到 DCU）----------------------------------
_ci_t="$(python3 - <<'PY' 2>/dev/null || true
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        p = torch.cuda.get_device_properties(0)
        print("name=" + str(getattr(p, "name", "")))
        u = getattr(p, "uuid", None)
        if u is not None:
            print("uuid=" + str(u))
        try:
            al = torch.cuda.get_arch_list()
            if al:
                print("gfx=" + ",".join(al))
        except Exception:
            pass
except Exception:
    pass
PY
)"
_ci_tn="$(printf '%s\n' "$_ci_t" | sed -n 's/^name=//p' | head -n 1)"
_ci_tu="$(printf '%s\n' "$_ci_t" | sed -n 's/^uuid=//p' | head -n 1)"
_ci_tg="$(printf '%s\n' "$_ci_t" | sed -n 's/^gfx=//p' | head -n 1)"
[ -z "$_ci_name" ] && [ -n "$_ci_tn" ] && _ci_name="$_ci_tn"
[ -z "$_ci_uid" ] && [ -n "$_ci_tu" ] && _ci_uid="$_ci_tu"
[ -z "$_ci_gfx" ] && [ -n "$_ci_tg" ] && _ci_gfx="$_ci_tg"

# --- 4) 主机标识（GPU 唯一标识取不到时的兜底）---------------------------------
_ci_host="$(hostname 2>/dev/null || echo unknown)"
_ci_boot="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"

if [ -n "$_ci_uid" ]; then
  CARD_ID="uid:${_ci_uid}"
  CARD_ID_SOURCE="gpu_unique_id"
else
  CARD_ID="host:${_ci_host}:boot:${_ci_boot}"
  CARD_ID_SOURCE="host_boot_id（GPU 唯一标识不可得，同卡未确认）"
fi
CARD_UID="$_ci_uid"
CARD_NAME="$_ci_name"
CARD_GFX="$_ci_gfx"
CARD_HOST="$_ci_host"
CARD_BOOT_ID="$_ci_boot"
CARD_ID_FILE="${CARD_ID_FILE:-$_ci_ws/outputs/CARD_ID.txt}"

# 落盘（供报告脚本比对）
if [ -n "${CARD_ID_FILE:-}" ]; then
  {
    echo "card_id=$CARD_ID"
    echo "card_id_source=$CARD_ID_SOURCE"
    echo "card_uid=$CARD_UID"
    echo "card_name=$CARD_NAME"
    echo "card_gfx=$CARD_GFX"
    echo "node_host=$CARD_HOST"
    echo "boot_id=$CARD_BOOT_ID"
    echo "captured_at=$(date -Is)"
  } > "$CARD_ID_FILE" 2>/dev/null || true
fi

export CARD_ID CARD_ID_SOURCE CARD_UID CARD_NAME CARD_GFX CARD_HOST CARD_BOOT_ID CARD_ID_FILE

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "card_id=$CARD_ID"
  echo "card_id_source=$CARD_ID_SOURCE"
  echo "card_uid=$CARD_UID"
  echo "card_name=$CARD_NAME"
  echo "card_gfx=$CARD_GFX"
  echo "node_host=$CARD_HOST"
  echo "boot_id=$CARD_BOOT_ID"
  echo "card_id_file=$CARD_ID_FILE"
fi
