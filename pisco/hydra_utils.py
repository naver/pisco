import hashlib

from omegaconf import OmegaConf


def _sanitize_override_dirname(override_dirname: object) -> str:
    """
    Make Hydra sweep subdir components stable and shorter:
    - For `model_name_or_path:<path>`, replace the value with a short hash.
    - Replace remaining `/` with `_` to keep paths filename-safe.
    """
    if not override_dirname:
        return ""

    raw = str(override_dirname).strip().strip("'\"")
    if raw == "":
        return ""

    # If this doesn't look like Hydra's override_dirname (no `--` separators),
    # treat it as a plain value/path (e.g. for exp_name) and just make it
    # filesystem-safe.
    if "--" not in raw:
        return raw.replace("/", "_").replace("\\", "").strip("_")

    def _split_items(s: str) -> list[str]:
        """
        Split on unescaped `--` (Hydra's `item_sep`), keeping values intact
        if they themselves contain `--` escaped with a backslash.
        """
        items: list[str] = []
        buf: list[str] = []

        i = 0
        while i < len(s):
            if i + 1 < len(s) and s[i] == "-" and s[i + 1] == "-":
                # Count consecutive backslashes immediately before this `--`.
                j = i - 1
                backslashes = 0
                while j >= 0 and s[j] == "\\":
                    backslashes += 1
                    j -= 1
                # If odd backslashes, treat `--` as escaped.
                if backslashes % 2 == 0:
                    chunk = "".join(buf)
                    if chunk != "":
                        items.append(chunk)
                    buf = []
                    i += 2
                    continue
            buf.append(s[i])
            i += 1

        chunk = "".join(buf)
        if chunk != "":
            items.append(chunk)
        return items

    out_parts: list[str] = []
    saw_model_name_or_path = False
    reading_model_value = False
    model_value_parts: list[str] = []

    def _is_plain_key(key_raw: str) -> bool:
        # Plain keys never contain Hydra escape backslashes.
        if "\\" in key_raw:
            return False
        key = key_raw.strip()
        if not key:
            return False
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        )
        return all(ch in allowed for ch in key)

    def _sanitize_kv(key_raw: str, val_raw: str) -> str:
        key = key_raw.replace("\\", "").strip()
        val = val_raw.strip().strip("'\"")
        val = val.replace("\\", "").replace("/", "_")
        return f"{key}:{val}"

    for item in _split_items(raw):
        if not reading_model_value:
            if ":" not in item:
                # Unknown segment, keep it but still make it safe for filesystem.
                out_parts.append(item.replace("/", "_").replace("\\", ""))
                continue

            key_raw, val_raw = item.split(":", 1)
            key_clean = key_raw.replace("\\", "").strip()
            if key_clean == "model_name_or_path":
                saw_model_name_or_path = True
                reading_model_value = True
                # Start capturing the full model path value across subsequent `--` segments.
                model_value_parts = [val_raw.strip()]
                continue

            out_parts.append(_sanitize_kv(key_raw, val_raw))
            continue

        # reading_model_value == True
        if ":" not in item:
            # Still part of the model path value.
            model_value_parts.append(item)
            continue

        next_key_raw, _ = item.split(":", 1)

        # If the next segment looks like a plain top-level key (not an escaped `key\:` from the model path),
        # we stop the model capture here and emit the hash.
        if _is_plain_key(next_key_raw):
            model_value = "--".join(model_value_parts).strip().strip("'\"")
            hashed = hashlib.sha1(model_value.encode("utf-8")).hexdigest()[:10]
            out_parts.append(f"model_name_or_path:{hashed}")
            reading_model_value = False

            # Now process the current item normally.
            if ":" in item:
                key_raw, val_raw = item.split(":", 1)
                out_parts.append(_sanitize_kv(key_raw, val_raw))
            continue

        # Otherwise, it is a `key\:`-like piece coming from inside the model path; keep capturing it.
        model_value_parts.append(item)

    if reading_model_value:
        # `model_name_or_path` was the last key in the override string.
        model_value = "--".join(model_value_parts).strip().strip("'\"")
        hashed = hashlib.sha1(model_value.encode("utf-8")).hexdigest()[:10]
        out_parts.append(f"model_name_or_path:{hashed}")
        reading_model_value = False

    if not out_parts:
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]

    if not saw_model_name_or_path:
        # If we can't identify the model path, use a stable hash of the whole string.
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]

    # Use '-' instead of '--' to avoid ambiguity with Hydra arg separation.
    return "-".join(out_parts)


def _path_after_user_dir(model_path: object, user_dir: object) -> str:
    """
    If model_path starts with user_dir, return the suffix after user_dir.
    Otherwise return model_path unchanged.
    """
    if not model_path:
        return ""

    full_path = str(model_path)
    if not user_dir:
        return full_path

    ud = str(user_dir).rstrip("/")
    if not ud:
        return full_path

    prefix = ud + "/"
    if full_path.startswith(prefix):
        suffix = full_path[len(prefix) :]
        return suffix or full_path

    return full_path


def register_resolvers() -> None:
    # Guard against re-registration in interactive environments.
    # OmegaConf throws if the resolver name already exists.
    for name, fn in [
        ("sanitize_override_dirname", _sanitize_override_dirname),
        ("path_after_user_dir", _path_after_user_dir),
    ]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            # Best-effort: if it's already registered, keep going.
            pass

