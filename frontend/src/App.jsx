import React, { useState, useCallback } from 'react';
import { Upload, Download, CheckCircle, AlertCircle, FileText } from 'lucide-react';
import Header from './components/Header';
import Guide from './components/Guide';
import SessionManager from './components/SessionManager';
import ProgressIndicator from './components/ProgressIndicator';
import ErrorBoundary from './components/ErrorBoundary';
import { useToast } from './components/Toast';

const SageInventoryApp = () => {
    const { showSuccess, showError, showInfo, ToastContainer } = useToast();
    
    // États pour la gestion des fichiers
    const [originalFile, setOriginalFile] = useState(null);
    const [completedFile, setCompletedFile] = useState(null);

    // États pour le statut des opérations
    const [uploadStatus, setUploadStatus] = useState('idle'); // 'idle', 'uploading', 'success', 'error'
    const [processStatus, setProcessStatus] = useState('idle'); // 'idle', 'processing', 'success', 'error'

    // Résultats des opérations
    const [uploadResult, setUploadResult] = useState(null);
    const [processResult, setProcessResult] = useState(null);

    // Gestion des erreurs
    const [error, setError] = useState('');
    
    // Étapes du processus
    const [currentStep, setCurrentStep] = useState(0);
    const [progressDetails, setProgressDetails] = useState('');
    
    const processSteps = [
        {
            title: "Import du fichier Sage X3",
            description: "Validation et traitement du fichier d'inventaire initial"
        },
        {
            title: "Génération du template",
            description: "Création du fichier Excel pour la saisie des quantités réelles"
        },
        {
            title: "Saisie des quantités",
            description: "Complétion du template avec les quantités réellement comptées"
        },
        {
            title: "Calcul des écarts",
            description: "Analyse des différences et répartition selon la stratégie FIFO"
        },
        {
            title: "Génération du fichier final",
            description: "Création du fichier CSV corrigé pour réimport dans Sage X3"
        }
    ];

    // URL de base de l'API
    const API_BASE_URL = 'http://localhost:5000/api';

    // Fonction helper pour créer un fichier mock
    const createMockFile = (filename, size = 0) => {
        return {
            name: filename,
            size: size,
            type: filename.endsWith('.xlsx') ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' : 'text/csv',
            lastModified: Date.now(),
            isMock: true // Flag pour identifier les fichiers mockés
        };
    };

    // Gestion du drag & drop
    const onDragOver = useCallback((e) => e.preventDefault(), []);

    const onDropOriginal = useCallback((e) => {
        e.preventDefault();
        const file = Array.from(e.dataTransfer.files)[0];
        if (file) {
            setOriginalFile(file);
            setError('');
        }
    }, []);

    const onDropCompleted = useCallback((e) => {
        e.preventDefault();
        const file = Array.from(e.dataTransfer.files)[0];
        if (file) {
            setCompletedFile(file);
            setError('');
        }
    }, []);

    // Traitement du fichier original Sage X3 (initial upload)
    const handleUploadFile = async () => {
        if (!originalFile) {
            setError('Veuillez sélectionner un fichier CSV Sage X3');
            showError('Veuillez sélectionner un fichier CSV Sage X3');
            return;
        }

        const formData = new FormData();
        formData.append('file', originalFile);

        setUploadStatus('uploading');
        setCurrentStep(0);
        setProgressDetails('Validation du format et traitement des données...');
        setError('');

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                setUploadStatus('success');
                setUploadResult(data);
                setCurrentStep(1);
                setProgressDetails('Template prêt à être téléchargé');
                showSuccess('Fichier traité avec succès !');
            } else {
                throw new Error(data.error || 'Erreur lors du traitement du fichier');
            }
        } catch (err) {
            setUploadStatus('error');
            setError(err.message);
            showError(err.message);
            setUploadResult(null);
            setProgressDetails('');
        }
    };

    // Génération du template Excel
    const handleDownloadTemplate = async () => {
        if (!uploadResult?.session_id) {
            setError('Aucune session active pour générer le template');
            showError('Aucune session active pour générer le template');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/download/template/${uploadResult.session_id}`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;

                const contentDisposition = response.headers.get('Content-Disposition');

                let filename = `inventaire_template_${uploadResult.session_id}.xlsx`;

                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename\*=(?:utf-8'')?([^;]+)|filename="([^"]+)"|filename=([^;]+)/i);
                    
                    if (filenameMatch) {
                        if (filenameMatch[1]) {
                            try {
                                filename = decodeURIComponent(filenameMatch[1]);
                            } catch (e) {
                                console.warn("Failed to decode URI component from filename*:", filenameMatch[1], e);
                                filename = filenameMatch[1];
                            }
                        } else if (filenameMatch[2]) {
                            filename = filenameMatch[2];
                        } else if (filenameMatch[3]) {
                            filename = filenameMatch[3].trim();
                        }
                    }
                }
                
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showSuccess('Template téléchargé avec succès !');
            } else {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erreur lors du téléchargement du template');
            }
        } catch (err) {
            setError(err.message);
            showError(err.message);
            console.error('Download template error:', err);
        }
    };

    // Traitement du fichier complété
    const handleProcessCompleted = async () => {
        if (!completedFile) {
            setError('Veuillez sélectionner le fichier Excel complété');
            showError('Veuillez sélectionner le fichier Excel complété');
            return;
        }
        if (!uploadResult?.session_id) {
            setError('Veuillez d\'abord importer et traiter le fichier original.');
            showError('Veuillez d\'abord importer et traiter le fichier original.');
            return;
        }

        const formData = new FormData();
        formData.append('file', completedFile);
        formData.append('session_id', uploadResult.session_id);

        setProcessStatus('processing');
        setCurrentStep(3);
        setProgressDetails('Calcul des écarts et répartition FIFO en cours...');
        setError('');

        try {
            const response = await fetch(`${API_BASE_URL}/process`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                setProcessStatus('success');
                setProcessResult(data);
                setCurrentStep(4);
                setProgressDetails('Fichier final généré et prêt au téléchargement');
                showSuccess('Traitement terminé avec succès !');
            } else {
                throw new Error(data.error || 'Erreur lors du calcul des écarts');
            }
        } catch (err) {
            setProcessStatus('error');
            setError(err.message);
            showError(err.message);
            setProcessResult(null);
            setProgressDetails('');
        }
    };

    // Fonction pour télécharger le fichier final corrigé
    const handleDownloadFinalFile = async () => {
        if (!uploadResult?.session_id) {
            setError('Aucun ID de session disponible pour télécharger le fichier final.');
            showError('Aucun ID de session disponible pour télécharger le fichier final.');
            return;
        }
        if (!processResult?.final_url) {
            setError('Le fichier final n\'a pas été généré ou son URL est manquante.');
            showError('Le fichier final n\'a pas été généré ou son URL est manquante.');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/download/final/${uploadResult.session_id}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                const contentDisposition = response.headers.get('Content-Disposition');
                
                let filename = `inventaire_corrige_${uploadResult.session_id}.csv`;
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/(?:filename\*=(?:UTF-8'')?([^;]+))|(?:filename="([^"]+)")|filename=([^;]+)/i);
                    if (filenameMatch) {
                        if (filenameMatch[1]) {
                            try {
                                filename = decodeURIComponent(filenameMatch[1]);
                            } catch (e) {
                                console.warn("Failed to decode URI component from filename* for final file:", filenameMatch[1], e);
                                filename = filenameMatch[1];
                            }
                        } else if (filenameMatch[2]) {
                            filename = filenameMatch[2];
                        } else if (filenameMatch[3]) {
                            filename = filenameMatch[3].trim();
                        }
                    }
                }
                a.download = filename;

                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                setError('');
                showSuccess('Fichier final téléchargé avec succès !');
            } else {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erreur lors du téléchargement du fichier final.');
            }
        } catch (err) {
            setError(err.message);
            showError(err.message);
            console.error('Download final file error:', err);
        }
    };

    // Réinitialisation
    const resetAll = () => {
        setOriginalFile(null);
        setCompletedFile(null);
        setUploadStatus('idle');
        setProcessStatus('idle');
        setUploadResult(null);
        setProcessResult(null);
        setError('');
        setCurrentStep(0);
        setProgressDetails('');
        showInfo('Session réinitialisée');
    };

    // Gestion de la sélection de session
    const handleSessionSelect = (session) => {
        // Restaurer l'état de l'application avec les données de la session
        try {
            // Réinitialiser d'abord l'état
            setOriginalFile(null);
            setCompletedFile(null);
            setError('');
            
            // Simuler un fichier original pour l'affichage
            const mockOriginalFile = {
                name: session.original_filename || 'Fichier original',
                size: 0
            };
            
            // Restaurer les résultats selon le statut de la session
            if (session.status === 'template_generated' || session.status === 'completed') {
                setOriginalFile(mockOriginalFile);
                setUploadStatus('success');
                setUploadResult({
                    session_id: session.id,
                    stats: {
                        nb_articles: session.stats?.nb_articles || 0,
                        total_quantity: session.stats?.total_quantity || 0,
                        nb_lots: session.stats?.nb_lots || 0
                    }
                });
                
                // Si la session est complètement terminée, restaurer aussi le résultat du traitement
                if (session.status === 'completed') {
                    setProcessStatus('success');
                    setCurrentStep(4);
                    setProgressDetails('Session terminée - Fichier final disponible');
                    setProcessResult({
                        final_url: `/api/download/final/${session.id}`,
                        stats: {
                            total_discrepancy: session.stats?.total_discrepancy || 0,
                            adjusted_items: session.stats?.adjusted_items_count || 0,
                            strategy_used: session.stats?.strategy_used || 'FIFO'
                        }
                    });
                    showSuccess(`Session ${session.id} reprise - Traitement terminé`);
                } else {
                    setProcessStatus('idle');
                    setProcessResult(null);
                    setCurrentStep(1);
                    setProgressDetails('Template disponible pour téléchargement');
                    showSuccess(`Session ${session.id} reprise - Template disponible`);
                }
            } else {
                // Pour les autres statuts, juste informer l'utilisateur
                setCurrentStep(0);
                setProgressDetails('');
                showInfo(`Session ${session.id} sélectionnée (statut: ${session.status})`);
            }
        } catch (error) {
            showError(`Erreur lors de la reprise de la session: ${error.message}`);
        }
    };

    return (
        <ErrorBoundary>
            <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 font-sans text-gray-800">
                <ToastContainer />
                
            {/* Header */}
            <Header />

            {/* Contenu principal */}
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Message d'erreur */}
                {error && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-6 animate-fade-in" role="alert">
                        <strong className="font-bold">Erreur : </strong>
                        <span className="block sm:inline">{error}</span>
                        <span className="absolute top-0 bottom-0 right-0 px-4 py-3">
                            <svg onClick={() => setError('')} className="fill-current h-6 w-6 text-red-500 cursor-pointer" role="button" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><title>Close</title><path d="M14.348 14.849a1.2 1.2 0 0 1-1.697 0L10 11.819l-2.651 3.029a1.2 1.2 0 1 1-1.697-1.697l2.758-3.15-2.759-3.152a1.2 1.2 0 1 1 1.697-1.697L10 8.183l2.651-3.031a1.2 1.2 0 1 1 1.697 1.697l-2.758 3.152 2.758 3.15a1.2 1.2 0 0 1 0 1.698z"/></svg>
                        </span>
                    </div>
                )}

                {/* Indicateur de progression */}
                {(uploadStatus !== 'idle' || processStatus !== 'idle') && (
                    <ProgressIndicator
                        steps={processSteps}
                        currentStep={currentStep}
                        status={uploadStatus === 'error' || processStatus === 'error' ? 'error' : 
                               uploadStatus === 'uploading' || processStatus === 'processing' ? 'processing' : 'idle'}
                        error={error}
                        details={progressDetails}
                    />
                )}

                {/* Étape 1: Import fichier original */}
                <div className={`bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200 transition-all duration-300 ${uploadStatus === 'success' ? 'opacity-75' : ''}`}>
                    <h2 className="text-xl font-semibold text-gray-900 mb-5 flex items-center">
                        <Upload className="h-6 w-6 mr-3 text-blue-600" />
                        1. Importation du fichier Sage X3
                    </h2>

                    {uploadStatus === 'idle' && (
                        <div className="space-y-5 animate-fade-in">
                            <div
                                className="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center hover:border-blue-500 transition-colors duration-200 cursor-pointer bg-gray-50"
                                onDragOver={onDragOver}
                                onDrop={onDropOriginal}
                                onClick={() => document.getElementById('original-file-input').click()}
                            >
                                <Upload className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                                <p className="text-xl font-medium text-gray-900 mb-2">
                                    Glissez-déposez votre fichier Sage X3 ici
                                </p>
                                <p className="text-sm text-gray-500 mb-4">
                                    ou cliquez pour sélectionner un fichier
                                </p>
                                <p className="text-xs text-gray-400">
                                    Formats acceptés: CSV, XLSX (format Sage X3 avec en-têtes E/L)
                                </p>
                            </div>

                            <input
                                id="original-file-input"
                                type="file"
                                accept=".csv,.xlsx"
                                onChange={(e) => setOriginalFile(e.target.files[0])}
                                className="hidden"
                            />

                            {originalFile && (
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center justify-between animate-fade-in">
                                    <div className="flex items-center space-x-3">
                                        <FileText className="h-8 w-8 text-blue-600" />
                                        <div>
                                            <p className="font-medium text-blue-900">{originalFile.name}</p>
                                            <p className="text-sm text-blue-700">
                                                {originalFile.isMock ? 'Session reprise' : `${(originalFile.size / 1024 / 1024).toFixed(2)} MB`}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex space-x-3">
                                        {!originalFile.isMock && (
                                            <>
                                        <button
                                            onClick={() => setOriginalFile(null)}
                                            className="px-4 py-2 text-sm text-blue-700 hover:text-blue-900 border border-blue-300 rounded-lg hover:bg-blue-100 transition-colors duration-200"
                                        >
                                            Annuler
                                        </button>
                                        <button
                                            onClick={handleUploadFile}
                                            className="px-5 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 font-medium shadow-md"
                                        >
                                            Traiter le fichier
                                        </button>
                                            </>
                                        )}
                                        {originalFile.isMock && (
                                            <div className="px-4 py-2 text-sm text-green-700 bg-green-100 rounded-lg">
                                                Session active
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {uploadStatus === 'uploading' && (
                        <div className="text-center py-12">
                            <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mb-4"></div>
                            <p className="text-lg text-gray-600">Traitement du fichier en cours...</p>
                        </div>
                    )}

                    {uploadStatus === 'success' && uploadResult && (
                        <div className="space-y-6 animate-fade-in">
                            <div className="bg-green-50 border border-green-200 rounded-xl p-5">
                                <div className="flex items-center mb-4">
                                    <CheckCircle className="h-6 w-6 text-green-600 mr-3" />
                                    <h3 className="font-semibold text-green-900 text-lg">Fichier traité avec succès !</h3>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
                                    <div className="bg-white rounded-lg p-4 border border-green-200 shadow-sm">
                                        <p className="text-sm text-green-700 font-medium">Articles traités</p>
                                        <p className="text-2xl font-bold text-green-900">{uploadResult.stats.nb_articles}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border border-green-200 shadow-sm">
                                        <p className="text-sm text-green-700 font-medium">Quantité totale</p>
                                        <p className="text-2xl font-bold text-green-900">{uploadResult.stats.total_quantity}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border border-green-200 shadow-sm">
                                        <p className="text-sm text-green-700 font-medium">Lots traités</p>
                                        <p className="text-2xl font-bold text-green-900">{uploadResult.stats.nb_lots}</p>
                                    </div>
                                </div>

                                <button
                                    onClick={handleDownloadTemplate}
                                    className="w-full bg-green-600 text-white py-3 px-4 rounded-xl hover:bg-green-700 transition-colors duration-200 font-semibold flex items-center justify-center shadow-md"
                                >
                                    <Download className="h-5 w-5 mr-2" />
                                    Télécharger
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Étape 2: Réimport fichier complété */}
                <div className={`bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200 transition-all duration-300 ${uploadStatus !== 'success' ? 'opacity-50 pointer-events-none' : ''}`}>
                    <h2 className="text-xl font-semibold text-gray-900 mb-5 flex items-center">
                        <Upload className="h-6 w-6 mr-3 text-purple-600" />
                        2. Réimportation du fichier complété
                    </h2>

                    {processStatus !== 'success' && (
                        <div className="space-y-5 animate-fade-in">
                            <div
                                className="border-2 border-dashed border-gray-300 rounded-xl p-10 text-center hover:border-purple-500 transition-colors duration-200 cursor-pointer bg-gray-50"
                                onDragOver={onDragOver}
                                onDrop={onDropCompleted}
                                onClick={() => document.getElementById('completed-file-input').click()}
                            >
                                <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                                <p className="text-xl font-medium text-gray-900 mb-2">
                                    Glissez-déposez le fichier Excel complété
                                </p>
                                <p className="text-sm text-gray-500 mb-4">
                                    ou cliquez pour sélectionner un fichier
                                </p>
                                <p className="text-xs text-gray-400">
                                    Format accepté: XLSX (template complété)
                                </p>
                            </div>

                            <input
                                id="completed-file-input"
                                type="file"
                                accept=".xlsx,.xls"
                                onChange={(e) => setCompletedFile(e.target.files[0])}
                                className="hidden"
                            />

                            {completedFile && (
                                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 flex items-center justify-between animate-fade-in">
                                    <div className="flex items-center space-x-3">
                                        <FileText className="h-8 w-8 text-purple-600" />
                                        <div>
                                            <p className="font-medium text-purple-900">{completedFile.name}</p>
                                            <p className="text-sm text-purple-700">
                                                {(completedFile.size / 1024 / 1024).toFixed(2)} MB
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex space-x-3">
                                        <button
                                            onClick={() => setCompletedFile(null)}
                                            className="px-4 py-2 text-sm text-purple-700 hover:text-purple-900 border border-purple-300 rounded-lg hover:bg-purple-100 transition-colors duration-200"
                                        >
                                            Annuler
                                        </button>
                                        <button
                                            onClick={handleProcessCompleted}
                                            className="px-5 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors duration-200 font-medium shadow-md"
                                        >
                                            Calculer
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {processStatus === 'processing' && (
                        <div className="text-center py-12">
                            <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-purple-600 mb-4"></div>
                            <p className="text-lg text-gray-600">Calcul des écarts et génération du fichier final...</p>
                        </div>
                    )}

                    {processStatus === 'success' && processResult && (
                        <div className="space-y-6 animate-fade-in">
                            <div className="bg-green-50 border border-green-200 rounded-xl p-5">
                                <div className="flex items-center mb-4">
                                    <CheckCircle className="h-6 w-6 text-green-600 mr-3" />
                                    <h3 className="font-semibold text-green-900 text-lg">Traitement terminé avec succès !</h3>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
                                    <div className="bg-white rounded-lg p-4 border border-green-200 shadow-sm">
                                        <p className="text-sm text-green-700 font-medium">Écart total</p>
                                        <p className="text-2xl font-bold text-green-900">{processResult.stats.total_discrepancy}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border border-green-200 shadow-sm">
                                        <p className="text-sm text-green-700 font-medium">Articles ajustés</p>
                                        <p className="text-2xl font-bold text-green-900">{processResult.stats.adjusted_items}</p>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border border-green-200 shadow-sm">
                                        <p className="text-sm text-green-700 font-medium">Fichier généré</p>
                                        <p className="text-sm font-mono text-green-900 break-all">Téléchargement ci-dessous</p>
                                    </div>
                                </div>

                                <div className="flex space-x-4">
                                    <button
                                        onClick={handleDownloadFinalFile}
                                        className="flex-1 bg-green-600 text-white py-3 px-4 rounded-xl hover:bg-green-700 transition-colors duration-200 font-semibold flex items-center justify-center shadow-md"
                                    >
                                        <Download className="h-5 w-5 mr-2" />
                                        Télécharger
                                    </button>
                                    <button
                                        onClick={resetAll}
                                        className="flex-1 border border-gray-300 text-gray-700 py-3 px-4 rounded-xl hover:bg-gray-100 transition-colors duration-200 font-medium shadow-sm"
                                    >
                                        Nouveau
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Guide du processus */}
                <Guide />
            </div>

                {/* Gestionnaire de sessions */}
                <SessionManager onSessionSelect={handleSessionSelect} />
        </div>
        </ErrorBoundary>
    );
};

export default SageInventoryApp;
