import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({
            error: error,
            errorInfo: errorInfo
        });
        
        // Log l'erreur (vous pouvez l'envoyer à un service de monitoring)
        console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    handleReload = () => {
        window.location.reload();
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-gradient-to-br from-red-50 via-white to-red-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-xl shadow-lg p-8 max-w-md w-full border border-red-200">
                        <div className="text-center">
                            <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
                            <h1 className="text-2xl font-bold text-gray-900 mb-2">
                                Oups ! Une erreur s'est produite
                            </h1>
                            <p className="text-gray-600 mb-6">
                                L'application a rencontré un problème inattendu. 
                                Veuillez recharger la page ou contacter le support si le problème persiste.
                            </p>
                            
                            {process.env.NODE_ENV === 'development' && this.state.error && (
                                <details className="text-left bg-gray-100 rounded-lg p-4 mb-6">
                                    <summary className="cursor-pointer font-medium text-gray-700 mb-2">
                                        Détails de l'erreur (développement)
                                    </summary>
                                    <pre className="text-xs text-red-600 overflow-auto">
                                        {this.state.error.toString()}
                                        {this.state.errorInfo.componentStack}
                                    </pre>
                                </details>
                            )}
                            
                            <button
                                onClick={this.handleReload}
                                className="bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 transition-colors duration-200 font-medium flex items-center justify-center mx-auto"
                            >
                                <RefreshCw className="h-5 w-5 mr-2" />
                                Recharger la page
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;