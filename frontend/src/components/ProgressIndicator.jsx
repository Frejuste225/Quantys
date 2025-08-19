import React from 'react';
import { CheckCircle, Clock, AlertCircle, Loader } from 'lucide-react';

const ProgressIndicator = ({ 
    steps, 
    currentStep, 
    status = 'idle', 
    error = null,
    details = null 
}) => {
    const getStepStatus = (stepIndex) => {
        if (stepIndex < currentStep) return 'completed';
        if (stepIndex === currentStep) {
            if (status === 'error') return 'error';
            if (status === 'processing' || status === 'uploading') return 'processing';
            return 'current';
        }
        return 'pending';
    };

    const getStepIcon = (stepStatus) => {
        switch (stepStatus) {
            case 'completed':
                return <CheckCircle className="h-6 w-6 text-green-600" />;
            case 'processing':
                return <Loader className="h-6 w-6 text-blue-600 animate-spin" />;
            case 'error':
                return <AlertCircle className="h-6 w-6 text-red-600" />;
            case 'current':
                return <Clock className="h-6 w-6 text-blue-600" />;
            default:
                return <div className="h-6 w-6 rounded-full border-2 border-gray-300" />;
        }
    };

    const getStepColors = (stepStatus) => {
        switch (stepStatus) {
            case 'completed':
                return 'bg-green-100 border-green-300 text-green-800';
            case 'processing':
                return 'bg-blue-100 border-blue-300 text-blue-800';
            case 'error':
                return 'bg-red-100 border-red-300 text-red-800';
            case 'current':
                return 'bg-blue-50 border-blue-300 text-blue-800';
            default:
                return 'bg-gray-50 border-gray-300 text-gray-600';
        }
    };

    return (
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Progression du traitement
            </h3>
            
            <div className="space-y-4">
                {steps.map((step, index) => {
                    const stepStatus = getStepStatus(index);
                    const isActive = index === currentStep;
                    
                    return (
                        <div key={index} className="flex items-start space-x-4">
                            <div className="flex-shrink-0 mt-1">
                                {getStepIcon(stepStatus)}
                            </div>
                            
                            <div className="flex-1 min-w-0">
                                <div className={`rounded-lg p-4 border-2 transition-all duration-200 ${getStepColors(stepStatus)}`}>
                                    <div className="flex items-center justify-between mb-2">
                                        <h4 className="font-semibold text-lg">
                                            {step.title}
                                        </h4>
                                        <span className="text-sm font-medium">
                                            Étape {index + 1}/{steps.length}
                                        </span>
                                    </div>
                                    
                                    <p className="text-sm mb-2">
                                        {step.description}
                                    </p>
                                    
                                    {isActive && details && (
                                        <div className="mt-3 p-3 bg-white bg-opacity-50 rounded-lg">
                                            <p className="text-sm font-medium text-gray-700">
                                                {details}
                                            </p>
                                        </div>
                                    )}
                                    
                                    {stepStatus === 'error' && error && (
                                        <div className="mt-3 p-3 bg-red-50 rounded-lg border border-red-200">
                                            <p className="text-sm text-red-700">
                                                <strong>Erreur:</strong> {error}
                                            </p>
                                        </div>
                                    )}
                                    
                                    {stepStatus === 'processing' && (
                                        <div className="mt-3">
                                            <div className="w-full bg-gray-200 rounded-full h-2">
                                                <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
            
            {/* Résumé global */}
            <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">
                        Progression globale
                    </span>
                    <span className="font-medium text-gray-900">
                        {Math.round((currentStep / (steps.length - 1)) * 100)}%
                    </span>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                    <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                        style={{width: `${Math.round((currentStep / (steps.length - 1)) * 100)}%`}}
                    ></div>
                </div>
            </div>
        </div>
    );
};

export default ProgressIndicator;