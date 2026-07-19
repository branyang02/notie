import { defaultUrlTransform } from "react-markdown";

/**
 * `tel:` links are safe and commonly used, but are not covered by
 * react-markdown's `defaultUrlTransform`.
 */
const TEL_URL = /^tel:/i;

/**
 * `data:image/*` URLs are safe when used as an image source (they cannot
 * execute script in an `<img>` context), but must never be allowed on
 * links, where `data:` URLs can be used for phishing or script execution.
 */
const DATA_IMAGE_URL = /^data:image\//i;

/**
 * URL transform for react-markdown that blocks dangerous URL schemes such as
 * `javascript:`, `vbscript:`, and `data:` (e.g. `[x](javascript:alert(1))`).
 *
 * Wraps react-markdown's `defaultUrlTransform`, which allows `http(s)`,
 * `mailto`, and relative/fragment/query URLs, and rejects everything else --
 * including URLs whose scheme is obfuscated with mixed case, whitespace, or
 * control characters. On top of that, this helper allows `tel:` links and
 * `data:image/*` URLs for image sources (`src` attributes) only.
 *
 * Matches react-markdown's `UrlTransform` contract:
 * `(url, key, node) => string` -- returning an empty string removes the URL.
 */
export function sanitizeUrl(url: string, key?: string): string {
    const transformed = defaultUrlTransform(url);
    if (transformed !== "") {
        return transformed;
    }

    if (TEL_URL.test(url)) {
        return url;
    }

    if (key === "src" && DATA_IMAGE_URL.test(url)) {
        return url;
    }

    return "";
}
