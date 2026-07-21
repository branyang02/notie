export interface RunCodeResponse {
    output: string;
    image: string;
}

export const DEFAULT_CODE_RUNNER_URL = "https://api.brandonyifanyang.com";
export const DEFAULT_RUN_CODE_TIMEOUT_MS = 60_000;

export interface RunCodeOptions {
    /** Base URL of the code-runner service. Defaults to the public notie runner. */
    baseUrl?: string;
    /** Optional external signal to abort the request (e.g. on unmount). */
    signal?: AbortSignal;
    /** Request timeout in milliseconds. Defaults to 60s. */
    timeoutMs?: number;
}

export const runCode = async (
    code: string,
    language: string,
    options: RunCodeOptions = {},
): Promise<RunCodeResponse> => {
    const {
        baseUrl = DEFAULT_CODE_RUNNER_URL,
        signal,
        timeoutMs = DEFAULT_RUN_CODE_TIMEOUT_MS,
    } = options;

    const controller = new AbortController();
    const onExternalAbort = () => controller.abort(signal?.reason);
    if (signal) {
        if (signal.aborted) {
            controller.abort(signal.reason);
        } else {
            signal.addEventListener("abort", onExternalAbort, { once: true });
        }
    }
    const timeoutId = setTimeout(() => {
        controller.abort(
            new Error(`Code execution timed out after ${timeoutMs}ms`),
        );
    }, timeoutMs);

    try {
        const response = await fetch(`${baseUrl}/api/coderunner`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ code, language }),
            signal: controller.signal,
        });

        if (!response.ok) {
            let message =
                `HTTP ${response.status} ${response.statusText}`.trim();
            const contentType = response.headers.get("content-type") ?? "";
            if (contentType.includes("application/json")) {
                try {
                    const errorData = await response.json();
                    if (
                        errorData &&
                        typeof errorData === "object" &&
                        typeof errorData.error === "string" &&
                        errorData.error
                    ) {
                        message = errorData.error;
                    }
                } catch {
                    // Body was not valid JSON despite the content type;
                    // fall back to the status-based message.
                }
            }
            throw new Error(message);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to run code:", error);
        throw error;
    } finally {
        clearTimeout(timeoutId);
        signal?.removeEventListener("abort", onExternalAbort);
    }
};
