export interface RunCodeResponse {
  output: string;
  image: string;
}

const baseUrl =
  import.meta.env.VITE_API_BASE_URL ||
  'https://yang-website-backend-c3338735a47f.herokuapp.com';

export const runCode = async (
  code: string,
  language: string,
): Promise<RunCodeResponse> => {
  try {
    const response = await fetch(`${baseUrl}/api/coderunner`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code, language }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Unknown error occurred');
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to run code:', error);
    throw error;
  }
};
