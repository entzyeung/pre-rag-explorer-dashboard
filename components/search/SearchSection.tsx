
import React, { useState, useEffect } from 'react';
import { VectorCollection, SearchResult } from '../../types';
import { generateQueryEmbedding } from '../../services/embeddingService';
import { cosineSimilarity, computeBM25 } from '../../utils/similarity';
import { Icons, CHUNKING_METHOD_LABELS } from '../../constants';
import CopyButton from '../common/CopyButton';
import { SAMPLE_PERSONAS } from '../../data/sampleQuestions';

interface SearchSectionProps {
  collections: VectorCollection[];
  loading: boolean;
}

const SearchSection: React.FC<SearchSectionProps> = ({ collections, loading: appLoading }) => {
  const [query, setQuery] = useState('');
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  
  // Persisted State: Retrieval Methods
  const [retrievalMethods, setRetrievalMethods] = useState<('dense' | 'sparse' | 'hybrid')[]>(() => {
    const saved = localStorage.getItem('rag_search_methods');
    return saved ? JSON.parse(saved) : ['dense'];
  });

  // Persisted State: Top K
  const [topK, setTopK] = useState(() => {
    const saved = localStorage.getItem('rag_search_topk');
    return saved ? parseInt(saved) : 5;
  });

  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedPersonaId, setSelectedPersonaId] = useState<string>('');

  const activePersona = SAMPLE_PERSONAS.find(p => p.id === selectedPersonaId);

  // Persistence Effects
  useEffect(() => {
    localStorage.setItem('rag_search_methods', JSON.stringify(retrievalMethods));
  }, [retrievalMethods]);

  useEffect(() => {
    localStorage.setItem('rag_search_topk', topK.toString());
  }, [topK]);

  // Bulk Actions
  const handleSelectAllCollections = () => setSelectedCollections(collections.map(c => c.id));
  const handleClearAllCollections = () => setSelectedCollections([]);
  
  const handleSelectAllRetrieval = () => setRetrievalMethods(['dense', 'sparse', 'hybrid']);
  const handleClearAllRetrieval = () => setRetrievalMethods([]);

  const handleSearch = async () => {
    if (!query) return;
    if (selectedCollections.length === 0) return alert("Select at least one collection.");
    
    setSearching(true);
    const allResults: SearchResult[] = [];

    try {
      const queryEmbedding = await generateQueryEmbedding(query);

      for (const colId of selectedCollections) {
        const col = collections.find(c => c.id === colId);
        if (!col) continue;

        const docTexts = col.chunks.map(c => c.text);
        
        // Retrieval Methods
        if (retrievalMethods.includes('dense')) {
          const scores = col.vectors.map(vec => cosineSimilarity(queryEmbedding, vec));
          scores.forEach((score, idx) => {
            allResults.push({
              chunk: col.chunks[idx],
              score,
              retrievalMethod: 'dense',
              collectionName: col.name,
              collectionId: col.id
            });
          });
        }

        if (retrievalMethods.includes('sparse')) {
          const scores = computeBM25(query, docTexts);
          // Normalize BM25 roughly for display (highly heuristic)
          const max = Math.max(...scores, 1);
          scores.forEach((s, idx) => {
            allResults.push({
              chunk: col.chunks[idx],
              score: s / max, 
              retrievalMethod: 'sparse',
              collectionName: col.name,
              collectionId: col.id
            });
          });
        }

        if (retrievalMethods.includes('hybrid')) {
            const denseScores = col.vectors.map(vec => cosineSimilarity(queryEmbedding, vec));
            const sparseScores = computeBM25(query, docTexts);
            const maxSparse = Math.max(...sparseScores, 1);
            
            denseScores.forEach((ds, idx) => {
                const ss = sparseScores[idx] / maxSparse;
                const hybridScore = (ds * 0.7) + (ss * 0.3);
                allResults.push({
                    chunk: col.chunks[idx],
                    score: hybridScore,
                    retrievalMethod: 'hybrid',
                    collectionName: col.name,
                    collectionId: col.id
                });
            });
        }
      }

      // Sort and take Top K for each method
      const finalResults = allResults
        .sort((a, b) => b.score - a.score)
        .slice(0, topK * retrievalMethods.length);
      
      setResults(finalResults);
    } catch (err) {
      console.error(err);
      alert("Search failed.");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <header>
        <h2 className="text-3xl font-bold text-slate-900 mb-2">Retrieval Explorer</h2>
        <p className="text-slate-500">Query your vector store and analyze how different retrieval methods perform on your indexed content.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">Search Target</h3>
              <div className="flex gap-2">
                 <button onClick={handleSelectAllCollections} className="text-[10px] font-bold text-blue-600 uppercase hover:text-blue-800 transition-colors">All</button>
                 <span className="text-slate-300">|</span>
                 <button onClick={handleClearAllCollections} className="text-[10px] font-bold text-slate-400 uppercase hover:text-slate-600 transition-colors">Clear</button>
              </div>
            </div>
            <div className="space-y-2 max-h-[250px] overflow-y-auto custom-scrollbar pr-1">
              {collections.map(col => (
                <label key={col.id} className="flex items-center gap-2 p-2 rounded hover:bg-slate-50 cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={selectedCollections.includes(col.id)}
                    onChange={() => setSelectedCollections(prev => prev.includes(col.id) ? prev.filter(x => x !== col.id) : [...prev, col.id])}
                    className="w-4 h-4 text-blue-600 rounded"
                  />
                  <span className="text-xs font-medium text-slate-700 truncate">{col.name}</span>
                </label>
              ))}
              {collections.length === 0 && <p className="text-xs text-slate-400 italic">No collections available.</p>}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
            <h3 className="text-sm font-bold text-slate-900 mb-3 uppercase tracking-wider">Parameters</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                   <label className="text-xs text-slate-500 block">Retrieval Method</label>
                   <div className="flex gap-2">
                     <button onClick={handleSelectAllRetrieval} className="text-[10px] font-bold text-blue-600 uppercase hover:text-blue-800 transition-colors">All</button>
                     <span className="text-slate-300">|</span>
                     <button onClick={handleClearAllRetrieval} className="text-[10px] font-bold text-slate-400 uppercase hover:text-slate-600 transition-colors">Clear</button>
                   </div>
                </div>
                <div className="space-y-1">
                  {['dense', 'sparse', 'hybrid'].map((m) => (
                    <label key={m} className="flex items-center gap-2 cursor-pointer">
                      <input 
                        type="checkbox" 
                        checked={retrievalMethods.includes(m as any)}
                        onChange={() => setRetrievalMethods(prev => prev.includes(m as any) ? prev.filter(x => x !== m) : [...prev, m as any])}
                        className="w-3 h-3 text-blue-600 rounded"
                      />
                      <span className="text-xs capitalize text-slate-700">{m} search</span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs text-slate-500">Top K Results</label>
                  <span className="text-xs font-bold text-blue-600">{topK}</span>
                </div>
                <input 
                  type="range" min="1" max="20" 
                  value={topK} 
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-6">
          {/* Persona Selection Area */}
          <div className="bg-gradient-to-r from-slate-50 to-white border border-slate-200 rounded-2xl p-5 shadow-sm">
            <div className="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <span className="w-5 h-5 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center text-xs">?</span>
                  Need inspiration?
                </h3>
                <p className="text-xs text-slate-500 mt-1">Select a persona to populate sample questions based on real-world scenarios.</p>
              </div>
              <select 
                value={selectedPersonaId}
                onChange={(e) => setSelectedPersonaId(e.target.value)}
                className="w-full md:w-auto px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-medium text-slate-700 focus:ring-2 focus:ring-blue-100 focus:border-blue-400 outline-none cursor-pointer hover:border-slate-300 transition-colors"
              >
                <option value="">-- Choose a User Persona --</option>
                {SAMPLE_PERSONAS.map(p => (
                  <option key={p.id} value={p.id}>{p.role}</option>
                ))}
              </select>
            </div>
            
            {activePersona && (
              <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="bg-indigo-50/50 p-3 rounded-xl mb-3 border border-indigo-100/50">
                   <p className="text-xs text-indigo-800 italic">{activePersona.description}</p>
                </div>
                <div className="flex flex-wrap gap-2 max-h-[200px] overflow-y-auto custom-scrollbar p-1">
                  {activePersona.questions.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => setQuery(q.text)}
                      className="text-left px-3 py-2 bg-white border border-slate-200 hover:border-blue-400 hover:bg-blue-50 hover:text-blue-700 rounded-lg text-xs font-medium text-slate-600 transition-all shadow-sm group"
                    >
                      <span className="block font-bold text-[10px] text-slate-400 uppercase tracking-wider mb-0.5 group-hover:text-blue-400">{q.focus}</span>
                      "{q.text}"
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="relative group">
            <textarea 
              placeholder="Ask a question about your documents..."
              className="w-full h-24 p-5 rounded-2xl border border-slate-200 bg-white focus:ring-4 focus:ring-blue-100 focus:border-blue-500 transition-all outline-none text-lg resize-none shadow-sm"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSearch())}
            />
            <button 
              onClick={handleSearch}
              disabled={searching || !query || selectedCollections.length === 0}
              className="absolute bottom-4 right-4 px-6 py-2 bg-blue-600 text-white font-bold rounded-xl shadow-lg hover:bg-blue-700 hover:scale-105 transition-all flex items-center gap-2 disabled:opacity-50 disabled:hover:scale-100"
            >
              {searching ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Icons.Search />
              )}
              Search
            </button>
          </div>

          <div className="space-y-4">
            {results.length > 0 ? (
              results.map((res, i) => (
                <div key={`${res.collectionId}-${res.chunk.id}-${i}`} className="bg-white border border-slate-200 rounded-2xl p-6 hover:border-blue-300 transition-all group shadow-sm">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-32 bg-slate-100 rounded-full overflow-hidden relative">
                         <div className={`absolute inset-y-0 left-0 bg-blue-500 transition-all duration-1000`} style={{ width: `${res.score * 100}%` }} />
                      </div>
                      <span className="text-xs font-bold text-blue-600">{(res.score * 100).toFixed(0)}% Match</span>
                    </div>
                    <div className="flex gap-2 items-center">
                       <CopyButton 
                         text={res.chunk.text} 
                         label="Copy" 
                         className="px-2 py-1 bg-slate-50 text-slate-500 border border-slate-200 rounded hover:bg-slate-100"
                       />
                       <span className="px-2 py-1 bg-indigo-50 text-indigo-600 text-[10px] font-bold rounded border border-indigo-100 uppercase">{res.retrievalMethod}</span>
                       <span className="px-2 py-1 bg-slate-100 text-slate-500 text-[10px] font-bold rounded border border-slate-200 uppercase">#{res.chunk.index}</span>
                    </div>
                  </div>
                  
                  <p className="text-slate-800 text-sm leading-relaxed mb-6 italic border-l-4 border-blue-500 pl-4 py-1">
                    "{res.chunk.text}"
                  </p>
                  
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-slate-400 font-medium">
                    <div className="flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 bg-slate-300 rounded-full" />
                      <span className="text-slate-500">Source:</span>
                      <span className="text-slate-700 font-bold">{res.chunk.sourceFileName}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 bg-slate-300 rounded-full" />
                      <span className="text-slate-500">Method:</span>
                      <span className="text-slate-700 font-bold">{CHUNKING_METHOD_LABELS[res.chunk.chunkMethod]}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 bg-slate-300 rounded-full" />
                      <span className="text-slate-500">Collection:</span>
                      <span className="text-slate-700 truncate max-w-[150px]">{res.collectionName}</span>
                    </div>
                  </div>
                </div>
              ))
            ) : query && !searching ? (
              <div className="text-center py-12 text-slate-400 italic">No matching chunks found. Try expanding your search criteria or selecting more collections.</div>
            ) : !searching && (
              <div className="text-center py-20 bg-slate-50 border-2 border-dashed border-slate-200 rounded-3xl">
                <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center mx-auto mb-3 shadow-sm text-slate-300">
                  <Icons.Search />
                </div>
                <p className="text-slate-400 font-medium">Enter a query and select collections to start exploring your RAG pipeline performance.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchSection;
