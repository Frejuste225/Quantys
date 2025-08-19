import { useState, useCallback } from 'react';

const API_BASE_URL = 'http://localhost:5000/api';

export const useApi = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const makeRequest = useCallback(async (endpoint, options = {}) => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                ...options,
                headers: {
                    ...options.headers,
                },
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Erreur HTTP: ${response.status}`);
            }

            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    const uploadFile = useCallback(async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        return makeRequest('/upload', {
            method: 'POST',
            body: formData,
        });
    }, [makeRequest]);

    const processFile = useCallback(async (file, sessionId, strategy = 'FIFO') => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', sessionId);
        formData.append('strategy', strategy);

        return makeRequest('/process', {
            method: 'POST',
            body: formData,
        });
    }, [makeRequest]);

    const downloadFile = useCallback(async (type, sessionId) => {
        const response = await fetch(`${API_BASE_URL}/download/${type}/${sessionId}`);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Erreur de téléchargement');
        }

        const blob = await response.blob();
        const contentDisposition = response.headers.get('Content-Disposition');
        
        let filename = `${type}_${sessionId}.${type === 'template' ? 'xlsx' : 'csv'}`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename\*=(?:UTF-8'')?([^;]+)|filename="([^"]+)"|filename=([^;]+)/i);
            if (filenameMatch) {
                if (filenameMatch[1]) {
                    try {
                        filename = decodeURIComponent(filenameMatch[1]);
                    } catch (e) {
                        filename = filenameMatch[1];
                    }
                } else if (filenameMatch[2]) {
                    filename = filenameMatch[2];
                } else if (filenameMatch[3]) {
                    filename = filenameMatch[3].trim();
                }
            }
        }

        // Créer et déclencher le téléchargement
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        return { filename, size: blob.size };
    }, []);

    const getSessions = useCallback(async () => {
        return makeRequest('/sessions');
    }, [makeRequest]);

    const deleteSession = useCallback(async (sessionId) => {
        return makeRequest(`/sessions/${sessionId}`, {
            method: 'DELETE'
        });
    }, [makeRequest]);

    const getHealth = useCallback(async () => {
        return makeRequest('/health');
    }, [makeRequest]);

    return {
        loading,
        error,
        uploadFile,
        processFile,
        downloadFile,
        getSessions,
        deleteSession,
        getHealth,
        clearError: () => setError(null)
    };
};