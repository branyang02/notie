export interface RunCodeResponse {
    output: string;
    image: string;
}

const baseUrl = "http://141.148.45.82:5000";

export const runCode = async (
    code: string,
    language: string,
): Promise<RunCodeResponse> => {
    try {
        const response = await fetch(`${baseUrl}/api/coderunner`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ code, language }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Unknown error occurred");
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to run code:", error);
        throw error;
    }
};
