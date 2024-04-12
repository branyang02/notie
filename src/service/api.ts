export interface RunCodeResponse {
  output: string;
  image: string;
}

export const runPythonCode = async (code: string): Promise<RunCodeResponse> => {
  const baseUrl =
    import.meta.env.VITE_API_BASE_URL ||
    'https://yang-website-backend-c3338735a47f.herokuapp.com';

  try {
    const response = await fetch(`${baseUrl}/api/run-python-code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to run code:', error);
    throw error; // Rethrow the error to be handled by the caller
  }
};

export const runCCode = async (code: string): Promise<RunCodeResponse> => {
  const baseUrl =
    import.meta.env.VITE_API_BASE_URL ||
    'https://yang-website-backend-c3338735a47f.herokuapp.com';

  try {
    const response = await fetch(`${baseUrl}/api/run-c-code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Failed to run code:', error);
    throw error; // Rethrow the error to be handled by the caller
  }
};
