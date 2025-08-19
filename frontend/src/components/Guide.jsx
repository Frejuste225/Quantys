import React from 'react';
import { Settings } from 'lucide-react';

const Guide = () => {
    return (
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <h3 className="text-xl font-semibold text-gray-900 mb-5 flex items-center">
                <Settings className="h-6 w-6 mr-3 text-blue-600" />
                    Guide du processus d'inventaire
            </h3>

            <div className="space-y-6">
                <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center border-2 border-blue-300">
                        <span className="text-lg font-bold text-blue-700">1</span>
                    </div>
                    <div>
                        <h4 className="font-semibold text-gray-900 text-lg">Importation du fichier Sage X3</h4>
                        <p className="text-sm text-gray-600">
                            Téléversez votre fichier d'inventaire Sage X3 au format CSV. Le fichier doit contenir les en-têtes E et L, suivis des données S. L'application validera la structure et préparera les données pour la saisie.
                        </p>
                    </div>
                </div>

                <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center border-2 border-purple-300">
                        <span className="text-lg font-bold text-purple-700">2</span>
                    </div>
                    <div>
                        <h4 className="font-semibold text-gray-900 text-lg">Saisie des quantités réelles</h4>
                        <p className="text-sm text-gray-600">
                            Téléchargez le template Excel généré, complétez les quantités réelles pour chaque article, puis réimportez le fichier complété. L'application calculera automatiquement les écarts.
                        </p>
                    </div>
                </div>

                <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0 w-10 h-10 bg-green-100 rounded-full flex items-center justify-center border-2 border-green-300">
                        <span className="text-lg font-bold text-green-700">3</span>
                    </div>
                    <div>
                        <h4 className="font-semibold text-gray-900 text-lg">Génération du fichier final</h4>
                        <p className="text-sm text-gray-600">
                            L'application répartira les écarts selon l'ancienneté des lots (FIFO) et générera un fichier CSV corrigé au format Sage X3, prêt à être réimporté dans votre système.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Guide;
