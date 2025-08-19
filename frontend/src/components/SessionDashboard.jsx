import React, { useState, useEffect } from 'react';
import { 
    Search, Filter, Calendar, FileText, Download, 
    Trash2, Archive, BarChart3, RefreshCw, Eye 
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import LoadingSpinner from './LoadingSpinner';

const SessionDashboard = ({ onSessionSelect, onClose }) => {
    const [sessions, setSessions] = useState([]);
    const [filteredSessions, setFilteredSessions] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [dateFilter, setDateFilter] = useState('all');
    const [sortBy, setSortBy] = useState('created_at');
    const [sortOrder, setSortOrder] = useState('desc');
    const [selectedSessions, setSelectedSessions] = useState([]);
    
    const { getSessions, downloadFile, deleteSession, loading } = useApi();

    useEffect(() => {
        loadSessions();
    }, []);

    useEffect(() => {
        filterAndSortSessions();
    }, [sessions, searchTerm, statusFilter, dateFilter, sortBy, sortOrder]);

    const loadSessions = async () => {
        try {
            const data = await getSessions();
            setSessions(data.sessions || []);
        } catch (error) {
            console.error('Erreur chargement sessions:', error);
        }
    };

    const filterAndSortSessions = () => {
        let filtered = [...sessions];

        // Filtrage par recherche
        if (searchTerm) {
            filtered = filtered.filter(session => 
                session.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                (session.original_filename || '').toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        // Filtrage par statut
        if (statusFilter !== 'all') {
            filtered = filtered.filter(session => session.status === statusFilter);
        }

        // Filtrage par date
        if (dateFilter !== 'all') {
            const now = new Date();
            const filterDate = new Date();
            
            switch (dateFilter) {
                case 'today':
                    filterDate.setHours(0, 0, 0, 0);
                    break;
                case 'week':
                    filterDate.setDate(now.getDate() - 7);
                    break;
                case 'month':
                    filterDate.setMonth(now.getMonth() - 1);
                    break;
            }
            
            filtered = filtered.filter(session => 
                new Date(session.created_at) >= filterDate
            );
        }

        // Tri
        filtered.sort((a, b) => {
            let aValue = a[sortBy];
            let bValue = b[sortBy];
            
            if (sortBy === 'created_at' || sortBy === 'updated_at') {
                aValue = new Date(aValue);
                bValue = new Date(bValue);
            }
            
            if (sortOrder === 'asc') {
                return aValue > bValue ? 1 : -1;
            } else {
                return aValue < bValue ? 1 : -1;
            }
        });

        setFilteredSessions(filtered);
    };

    const handleBulkAction = async (action) => {
        if (selectedSessions.length === 0) return;

        try {
            switch (action) {
                case 'delete':
                    for (const sessionId of selectedSessions) {
                        await deleteSession(sessionId);
                    }
                    setSessions(sessions.filter(s => !selectedSessions.includes(s.id)));
                    break;
                case 'archive':
                    // Implémentation de l'archivage
                    console.log('Archive sessions:', selectedSessions);
                    break;
            }
            setSelectedSessions([]);
        } catch (error) {
            console.error('Erreur action groupée:', error);
        }
    };

    const getStatusStats = () => {
        const stats = sessions.reduce((acc, session) => {
            acc[session.status] = (acc[session.status] || 0) + 1;
            return acc;
        }, {});
        
        return {
            total: sessions.length,
            ...stats
        };
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleString('fr-FR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getStatusColor = (status) => {
        const colors = {
            'template_generated': 'bg-blue-100 text-blue-800 border-blue-200',
            'completed': 'bg-green-100 text-green-800 border-green-200',
            'error': 'bg-red-100 text-red-800 border-red-200',
            'processing': 'bg-yellow-100 text-yellow-800 border-yellow-200',
            'uploading': 'bg-gray-100 text-gray-800 border-gray-200',
            'archived': 'bg-purple-100 text-purple-800 border-purple-200'
        };
        return colors[status] || 'bg-gray-100 text-gray-800 border-gray-200';
    };

    const stats = getStatusStats();

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-[55] flex items-center justify-center p-4"
             onClick={(e) => {
                 if (e.target === e.currentTarget) {
                     onClose();
                 }
             }}>
            <div className="bg-white rounded-xl shadow-2xl max-w-7xl w-full max-h-[90vh] overflow-hidden">
                {/* En-tête */}
                <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-2xl font-bold flex items-center">
                            <BarChart3 className="h-7 w-7 mr-3" />
                            Tableau de bord des sessions
                        </h2>
                        <div className="flex items-center space-x-3">
                            <button
                                onClick={loadSessions}
                                disabled={loading}
                                className="p-2 hover:bg-blue-700 rounded-lg transition-colors duration-200"
                            >
                                <RefreshCw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
                            </button>
                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-blue-700 rounded-lg transition-colors duration-200"
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                    
                    {/* Statistiques */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div className="bg-white bg-opacity-20 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold">{stats.total}</div>
                            <div className="text-sm opacity-90">Total</div>
                        </div>
                        <div className="bg-white bg-opacity-20 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-green-200">{stats.completed || 0}</div>
                            <div className="text-sm opacity-90">Terminées</div>
                        </div>
                        <div className="bg-white bg-opacity-20 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-blue-200">{stats.template_generated || 0}</div>
                            <div className="text-sm opacity-90">En cours</div>
                        </div>
                        <div className="bg-white bg-opacity-20 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-red-200">{stats.error || 0}</div>
                            <div className="text-sm opacity-90">Erreurs</div>
                        </div>
                        <div className="bg-white bg-opacity-20 rounded-lg p-3 text-center">
                            <div className="text-2xl font-bold text-purple-200">{stats.archived || 0}</div>
                            <div className="text-sm opacity-90">Archivées</div>
                        </div>
                    </div>
                </div>

                {/* Filtres et recherche */}
                <div className="p-6 border-b border-gray-200 bg-gray-50">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        {/* Recherche */}
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Rechercher par ID ou nom de fichier..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            />
                        </div>

                        {/* Filtre par statut */}
                        <select
                            value={statusFilter}
                            onChange={(e) => setStatusFilter(e.target.value)}
                            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                            <option value="all">Tous les statuts</option>
                            <option value="template_generated">Template généré</option>
                            <option value="completed">Terminé</option>
                            <option value="error">Erreur</option>
                            <option value="processing">En cours</option>
                            <option value="archived">Archivé</option>
                        </select>

                        {/* Filtre par date */}
                        <select
                            value={dateFilter}
                            onChange={(e) => setDateFilter(e.target.value)}
                            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                            <option value="all">Toutes les dates</option>
                            <option value="today">Aujourd'hui</option>
                            <option value="week">Cette semaine</option>
                            <option value="month">Ce mois</option>
                        </select>

                        {/* Tri */}
                        <select
                            value={`${sortBy}-${sortOrder}`}
                            onChange={(e) => {
                                const [field, order] = e.target.value.split('-');
                                setSortBy(field);
                                setSortOrder(order);
                            }}
                            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                            <option value="created_at-desc">Plus récent</option>
                            <option value="created_at-asc">Plus ancien</option>
                            <option value="id-asc">ID croissant</option>
                            <option value="id-desc">ID décroissant</option>
                        </select>
                    </div>

                    {/* Actions groupées */}
                    {selectedSessions.length > 0 && (
                        <div className="mt-4 flex items-center space-x-3">
                            <span className="text-sm text-gray-600">
                                {selectedSessions.length} session(s) sélectionnée(s)
                            </span>
                            <button
                                onClick={() => handleBulkAction('delete')}
                                className="px-3 py-1 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors duration-200 text-sm"
                            >
                                <Trash2 className="h-4 w-4 inline mr-1" />
                                Supprimer
                            </button>
                            <button
                                onClick={() => handleBulkAction('archive')}
                                className="px-3 py-1 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors duration-200 text-sm"
                            >
                                <Archive className="h-4 w-4 inline mr-1" />
                                Archiver
                            </button>
                        </div>
                    )}
                </div>

                {/* Liste des sessions */}
                <div className="p-6 overflow-y-auto max-h-[calc(90vh-400px)]">
                    {loading ? (
                        <LoadingSpinner message="Chargement des sessions..." />
                    ) : filteredSessions.length === 0 ? (
                        <div className="text-center py-12">
                            <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                            <p className="text-gray-600">
                                {searchTerm || statusFilter !== 'all' || dateFilter !== 'all' 
                                    ? 'Aucune session ne correspond aux critères de recherche'
                                    : 'Aucune session trouvée'
                                }
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {filteredSessions.map((session) => (
                                <div
                                    key={session.id}
                                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-all duration-200 bg-white"
                                >
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                checked={selectedSessions.includes(session.id)}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        setSelectedSessions([...selectedSessions, session.id]);
                                                    } else {
                                                        setSelectedSessions(selectedSessions.filter(id => id !== session.id));
                                                    }
                                                }}
                                                className="h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                                            />
                                            <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">
                                                {session.id}
                                            </span>
                                            <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getStatusColor(session.status)}`}>
                                                {session.status}
                                            </span>
                                        </div>
                                        <div className="text-sm text-gray-500">
                                            {formatDate(session.created_at)}
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                                        <div>
                                            <p className="text-sm text-gray-600">Fichier</p>
                                            <p className="font-medium truncate" title={session.original_filename}>
                                                {session.original_filename || 'N/A'}
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-600">Articles</p>
                                            <p className="font-medium">{session.stats?.nb_articles || 0}</p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-600">Quantité</p>
                                            <p className="font-medium">{session.stats?.total_quantity || 0}</p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-600">Écart total</p>
                                            <p className="font-medium">{session.stats?.total_discrepancy || 0}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={() => onSessionSelect(session)}
                                                className="flex items-center px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors duration-200 text-sm font-medium"
                                            >
                                                <Eye className="h-4 w-4 mr-1" />
                                                Ouvrir
                                            </button>
                                            
                                            {session.status === 'template_generated' && (
                                                <button
                                                    onClick={() => downloadFile('template', session.id)}
                                                    className="flex items-center px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors duration-200 text-sm"
                                                >
                                                    <Download className="h-4 w-4 mr-1" />
                                                    Template
                                                </button>
                                            )}
                                            
                                            {session.status === 'completed' && (
                                                <button
                                                    onClick={() => downloadFile('final', session.id)}
                                                    className="flex items-center px-3 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors duration-200 text-sm"
                                                >
                                                    <Download className="h-4 w-4 mr-1" />
                                                    Final
                                                </button>
                                            )}
                                        </div>
                                        
                                        <button
                                            onClick={() => deleteSession(session.id).then(() => loadSessions())}
                                            className="flex items-center px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors duration-200 text-sm"
                                        >
                                            <Trash2 className="h-4 w-4 mr-1" />
                                            Supprimer
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SessionDashboard;