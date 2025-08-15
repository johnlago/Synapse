#!/usr/bin/env python3
import os
import logging
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
import chromadb
import ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-mcp-server")

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434") 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# Initialize FastMCP server
mcp = FastMCP("synapse")

# Global clients
chroma_client = None
collection = None

# Search history storage (in-memory for now)
search_history = []
MAX_HISTORY_SIZE = 50

def highlight_text(text: str, query_terms: List[str], max_length: int = 500) -> str:
    """Highlight query terms in text and return snippet"""
    if not query_terms:
        return text[:max_length] + ('...' if len(text) > max_length else '')
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    highlighted_text = text
    
    # Find best snippet containing query terms
    best_start = 0
    best_score = 0
    
    # Look for snippet with most query terms
    for i in range(0, len(text) - max_length + 1, 50):
        snippet = text_lower[i:i + max_length]
        score = sum(1 for term in query_terms if term.lower() in snippet)
        if score > best_score:
            best_score = score
            best_start = i
    
    # Extract best snippet
    snippet = text[best_start:best_start + max_length]
    
    # Highlight terms (simple markdown bold)
    for term in query_terms:
        if len(term) > 2:  # Only highlight meaningful terms
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            snippet = pattern.sub(f"**{term}**", snippet)
    
    return snippet + ('...' if len(text) > max_length else '')

def add_to_search_history(query: str, result_count: int):
    """Add search query to history"""
    global search_history
    
    # Avoid duplicates
    search_history = [item for item in search_history if item['query'] != query]
    
    # Add new entry
    search_history.insert(0, {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'result_count': result_count
    })
    
    # Limit history size
    if len(search_history) > MAX_HISTORY_SIZE:
        search_history = search_history[:MAX_HISTORY_SIZE]

def extract_keywords_from_documents() -> List[str]:
    """Extract common keywords from documents for auto-suggest"""
    try:
        # Get a sample of documents
        results = collection.get(limit=100, include=["documents", "metadatas"])
        
        if not results['documents']:
            return []
        
        # Extract keywords from documents
        word_freq = {}
        for doc in results['documents']:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most common words (excluding common stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'with'}
        
        filtered_words = [(word, freq) for word, freq in word_freq.items() 
                         if word not in stopwords and freq > 2]
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:50]]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

async def init_clients():
    """Initialize Chroma client and collection"""
    global chroma_client, collection
    if not chroma_client:
        try:
            chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST.split(':')[0], 
                port=int(CHROMA_HOST.split(':')[1])
            )
            collection = chroma_client.get_or_create_collection(
                "documents",
                metadata={"description": "Local RAG document collection", "hnsw:space": "cosine"}
            )
            logger.info("Initialized Chroma client and collection")
        except Exception as e:
            logger.error(f"Error connecting to Chroma: {e}")
            raise

async def get_embeddings(text: str) -> List[float]:
    """Get embeddings from Ollama"""
    try:
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise

@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.001
) -> str:
    """Search through indexed documents using semantic similarity
    
    Args:
        query: The search query
        limit: Maximum number of results to return (default: 5)
        min_similarity: Minimum similarity score from 0.0 to 1.0 (default: 0.001)
    
    Returns:
        Formatted search results with similarity scores
    """
    await init_clients()
    
    try:
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Search in Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Format results with highlighting
        formatted_results = []
        query_terms = query.split()
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # Convert cosine distance to similarity
            similarity = max(0, 1 - distance)
            if similarity >= min_similarity:
                # Highlight query terms in content
                highlighted_content = highlight_text(doc, query_terms)
                
                formatted_results.append(
                    f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                    f"**Source:** {metadata.get('file_name', 'Unknown')}\n"
                    f"**Content:** {highlighted_content}\n"
                )
        
        if not formatted_results:
            return f"No documents found with similarity >= {min_similarity}"
        
        # Add to search history
        add_to_search_history(query, len(formatted_results))
        
        return f"Found {len(formatted_results)} relevant documents:\n\n" + "\n---\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error searching documents: {e}"

@mcp.tool()
async def get_document_info(
    file_name: str = None
) -> str:
    """Get information about indexed documents
    
    Args:
        file_name: Optional specific file name to get info for
    
    Returns:
        Document information or overall collection statistics
    """
    await init_clients()
    
    try:
        if file_name:
            # Get info for specific file
            results = collection.get(
                where={"file_name": file_name},
                include=["metadatas"]
            )
            if not results['ids']:
                return f"No document found with name: {file_name}"
            
            return (
                f"Document: {file_name}\n"
                f"Chunks: {len(results['ids'])}\n"
                f"Last processed: {results['metadatas'][0].get('processed_at', 'Unknown')}"
            )
        else:
            # Get overall collection info
            count = collection.count()
            
            # Get unique files
            all_metadata = collection.get(include=["metadatas"])
            unique_files = set()
            for metadata in all_metadata['metadatas']:
                unique_files.add(metadata.get('file_name', 'Unknown'))
            
            return (
                f"RAG Database Status:\n"
                f"Total chunks: {count}\n"
                f"Unique documents: {len(unique_files)}\n"
                f"Documents: {', '.join(sorted(unique_files))}"
            )
            
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        return f"Error getting document info: {e}"

@mcp.tool()
async def search_by_source(
    query: str,
    source_filter: str,
    limit: int = 5
) -> str:
    """Search within specific document sources
    
    Args:
        query: The search query
        source_filter: Filter by source file name or path (supports partial matching)
        limit: Maximum number of results (default: 5)
    
    Returns:
        Formatted search results from the filtered source
    """
    await init_clients()
    
    try:
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Get all results first, then filter manually since ChromaDB doesn't support $contains
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 10,  # Get more to filter
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return f"No documents found matching the query"
        
        # Filter results by source
        filtered_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            file_name = metadata.get('file_name', '')
            source_path = metadata.get('source', '')
            # Check if source_filter matches filename or full path
            if (source_filter.lower() in file_name.lower() or 
                source_filter.lower() in source_path.lower()):
                filtered_results.append((doc, metadata, distance))
                if len(filtered_results) >= limit:
                    break
        
        if not filtered_results:
            return f"No documents found containing '{source_filter}' in filename or path"
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(filtered_results):
            # Convert cosine distance to similarity
            similarity = max(0, 1 - distance)
            formatted_results.append(
                f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')}\n"
                f"**Path:** {metadata.get('source', 'Unknown')}\n"
                f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
            )
        
        return (
            f"Found {len(formatted_results)} results containing '{source_filter}':\n\n" +
            "\n---\n".join(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error searching by source: {e}")
        return f"Error searching by source: {e}"

@mcp.tool()
async def search_by_type(
    query: str,
    file_types: str,
    limit: int = 5,
    min_similarity: float = 0.001
) -> str:
    """Search documents filtered by file type/extension
    
    Args:
        query: The search query
        file_types: Comma-separated list of file extensions (e.g., 'pdf,docx' or '.md,.txt')
        limit: Maximum number of results (default: 5)
        min_similarity: Minimum similarity score (default: 0.001)
    
    Returns:
        Formatted search results filtered by file type
    """
    await init_clients()
    
    try:
        # Parse file types
        types = [t.strip().lower().lstrip('.') for t in file_types.split(',')]
        
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Get all results first, then filter by type
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 10,  # Get more to filter
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Filter by file type
        filtered_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            file_name = metadata.get('file_name', '')
            file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
            
            if file_ext in types:
                similarity = max(0, 1 - distance)
                if similarity >= min_similarity:
                    filtered_results.append((doc, metadata, distance, similarity))
                    if len(filtered_results) >= limit:
                        break
        
        if not filtered_results:
            return f"No {', '.join(types)} documents found with similarity >= {min_similarity}"
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance, similarity) in enumerate(filtered_results):
            formatted_results.append(
                f"**Result {i+1}** (similarity: {similarity:.3f})\\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')}\\n"
                f"**Type:** {metadata.get('file_name', '').split('.')[-1].upper() if '.' in metadata.get('file_name', '') else 'Unknown'}\\n"
                f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\\n"
            )
        
        return f"Found {len(filtered_results)} {', '.join(types)} documents:\\n\\n" + "\\n---\\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error searching by type: {e}")
        return f"Error searching by type: {e}"

@mcp.tool()
async def search_by_date(
    query: str,
    days_back: int = 30,
    limit: int = 5,
    min_similarity: float = 0.001
) -> str:
    """Search documents filtered by processing date (recency)
    
    Args:
        query: The search query
        days_back: Number of days back to include documents (default: 30)
        limit: Maximum number of results (default: 5)
        min_similarity: Minimum similarity score (default: 0.001)
    
    Returns:
        Formatted search results filtered by date
    """
    await init_clients()
    
    try:
        # Calculate cutoff timestamp
        cutoff_time = (datetime.now() - timedelta(days=days_back)).timestamp()
        
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Get all results first, then filter by date
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 10,  # Get more to filter
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Filter by date
        filtered_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            processed_at = metadata.get('processed_at', 0)
            
            if processed_at >= cutoff_time:
                similarity = max(0, 1 - distance)
                if similarity >= min_similarity:
                    filtered_results.append((doc, metadata, distance, similarity))
                    if len(filtered_results) >= limit:
                        break
        
        if not filtered_results:
            return f"No documents processed in the last {days_back} days with similarity >= {min_similarity}"
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance, similarity) in enumerate(filtered_results):
            processed_at = metadata.get('processed_at', 0)
            processed_date = datetime.fromtimestamp(processed_at).strftime('%Y-%m-%d %H:%M') if processed_at else 'Unknown'
            
            formatted_results.append(
                f"**Result {i+1}** (similarity: {similarity:.3f})\\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')}\\n"
                f"**Processed:** {processed_date}\\n"
                f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\\n"
            )
        
        return f"Found {len(filtered_results)} documents from last {days_back} days:\\n\\n" + "\\n---\\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error searching by date: {e}")
        return f"Error searching by date: {e}"

@mcp.tool()
async def advanced_search(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.001,
    file_types: Optional[str] = None,
    days_back: Optional[int] = None,
    source_filter: Optional[str] = None,
    expand_query: bool = True
) -> str:
    """Advanced search with multiple filters and query expansion
    
    Args:
        query: The search query
        limit: Maximum number of results (default: 5)
        min_similarity: Minimum similarity score (default: 0.001)
        file_types: Optional comma-separated file extensions (e.g., 'pdf,docx')
        days_back: Optional number of days back to include documents
        source_filter: Optional filter by filename or path
        expand_query: Whether to expand query with synonyms (default: True)
    
    Returns:
        Formatted search results with applied filters
    """
    await init_clients()
    
    try:
        # Expand query if requested
        search_query = query
        if expand_query:
            # Simple query expansion with common technical synonyms
            synonyms = {
                'api': ['endpoint', 'interface', 'service'],
                'function': ['method', 'procedure', 'routine'],
                'error': ['exception', 'failure', 'bug'],
                'config': ['configuration', 'settings', 'setup'],
                'database': ['db', 'storage', 'data'],
                'auth': ['authentication', 'login', 'security'],
                'deploy': ['deployment', 'release', 'publish'],
                'test': ['testing', 'validation', 'verification']
            }
            
            expanded_terms = []
            for word in query.lower().split():
                if word in synonyms:
                    expanded_terms.extend(synonyms[word])
            
            if expanded_terms:
                search_query = f"{query} {' '.join(expanded_terms[:3])}"  # Limit expansion
        
        # Get query embedding
        query_embedding = await get_embeddings(search_query)
        
        # Get more results for filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 20,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Apply filters
        filtered_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = max(0, 1 - distance)
            if similarity < min_similarity:
                continue
            
            # File type filter
            if file_types:
                types = [t.strip().lower().lstrip('.') for t in file_types.split(',')]
                file_name = metadata.get('file_name', '')
                file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
                if file_ext not in types:
                    continue
            
            # Date filter
            if days_back:
                cutoff_time = (datetime.now() - timedelta(days=days_back)).timestamp()
                processed_at = metadata.get('processed_at', 0)
                if processed_at < cutoff_time:
                    continue
            
            # Source filter
            if source_filter:
                file_name = metadata.get('file_name', '')
                source_path = metadata.get('source', '')
                if not (source_filter.lower() in file_name.lower() or 
                       source_filter.lower() in source_path.lower()):
                    continue
            
            filtered_results.append((doc, metadata, distance, similarity))
            if len(filtered_results) >= limit:
                break
        
        if not filtered_results:
            return "No documents found matching your search criteria."
        
        # Format results with enhanced information
        formatted_results = []
        for i, (doc, metadata, distance, similarity) in enumerate(filtered_results):
            processed_at = metadata.get('processed_at', 0)
            processed_date = datetime.fromtimestamp(processed_at).strftime('%Y-%m-%d') if processed_at else 'Unknown'
            file_type = metadata.get('file_name', '').split('.')[-1].upper() if '.' in metadata.get('file_name', '') else 'Unknown'
            
            # Highlight query terms in content (simple highlighting)
            content = doc[:500]
            if expand_query and search_query != query:
                content += f"\\n*[Query expanded: {search_query}]*"
            
            formatted_results.append(
                f"**Result {i+1}** (similarity: {similarity:.3f})\\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')} ({file_type})\\n"
                f"**Processed:** {processed_date}\\n"
                f"**Content:** {content}{'...' if len(doc) > 500 else ''}\\n"
            )
        
        filter_info = []
        if file_types:
            filter_info.append(f"types: {file_types}")
        if days_back:
            filter_info.append(f"last {days_back} days")
        if source_filter:
            filter_info.append(f"source: {source_filter}")
        
        filter_text = f" (filters: {', '.join(filter_info)})" if filter_info else ""
        expansion_text = f" (expanded from: {query})" if expand_query and search_query != query else ""
        
        return f"Found {len(filtered_results)} documents{filter_text}{expansion_text}:\\n\\n" + "\\n---\\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        return f"Error in advanced search: {e}"

@mcp.tool()
async def search_with_clustering(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.001,
    cluster_threshold: float = 0.8
) -> str:
    """Search documents with result clustering by similarity
    
    Args:
        query: The search query
        limit: Maximum number of results (default: 10)
        min_similarity: Minimum similarity score (default: 0.001)
        cluster_threshold: Similarity threshold for clustering results (default: 0.8)
    
    Returns:
        Formatted search results grouped by similarity clusters
    """
    await init_clients()
    
    try:
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Get more results for clustering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Filter by similarity
        valid_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = max(0, 1 - distance)
            if similarity >= min_similarity:
                valid_results.append((doc, metadata, distance, similarity))
        
        if not valid_results:
            return f"No documents found with similarity >= {min_similarity}"
        
        # Simple clustering by file/source grouping
        clusters = {}
        for doc, metadata, distance, similarity in valid_results[:limit]:
            source = metadata.get('file_name', 'Unknown')
            if source not in clusters:
                clusters[source] = []
            clusters[source].append((doc, metadata, distance, similarity))
        
        # Format clustered results
        formatted_output = []
        cluster_num = 1
        
        for source, docs in clusters.items():
            # Sort by similarity within cluster
            docs.sort(key=lambda x: x[3], reverse=True)
            
            formatted_output.append(f"\\n**Cluster {cluster_num}: {source}** ({len(docs)} results)")
            
            for i, (doc, metadata, distance, similarity) in enumerate(docs):
                processed_at = metadata.get('processed_at', 0)
                processed_date = datetime.fromtimestamp(processed_at).strftime('%Y-%m-%d') if processed_at else 'Unknown'
                
                formatted_output.append(
                    f"  **{cluster_num}.{i+1}** (similarity: {similarity:.3f}) - {processed_date}\\n"
                    f"  {doc[:300]}{'...' if len(doc) > 300 else ''}\\n"
                )
            
            cluster_num += 1
        
        return f"Found {len(valid_results)} documents in {len(clusters)} clusters:\\n" + "\\n".join(formatted_output)
        
    except Exception as e:
        logger.error(f"Error in clustered search: {e}")
        return f"Error in clustered search: {e}"

@mcp.tool()
async def follow_up_search(
    previous_results: str,
    follow_up_query: str,
    limit: int = 5
) -> str:
    """Perform a follow-up search based on previous results
    
    Args:
        previous_results: Text from previous search results to use as context
        follow_up_query: New search query to refine results
        limit: Maximum number of results (default: 5)
    
    Returns:
        Formatted search results refined by follow-up query
    """
    await init_clients()
    
    try:
        # Combine previous context with new query for enhanced search
        combined_query = f"{follow_up_query} {previous_results[:500]}"
        
        # Get query embedding
        query_embedding = await get_embeddings(combined_query)
        
        # Search with combined query
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found for follow-up search."
        
        # Get embeddings for follow-up query alone to compare
        followup_embedding = await get_embeddings(follow_up_query)
        
        # Re-rank results based on follow-up query relevance
        reranked_results = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Calculate similarity to follow-up query specifically
            followup_similarity = max(0, 1 - distance)  # Using combined for now
            reranked_results.append((doc, metadata, distance, followup_similarity))
        
        # Sort by follow-up relevance and take top results
        reranked_results.sort(key=lambda x: x[3], reverse=True)
        final_results = reranked_results[:limit]
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance, similarity) in enumerate(final_results):
            formatted_results.append(
                f"**Follow-up Result {i+1}** (relevance: {similarity:.3f})\\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')}\\n"
                f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\\n"
            )
        
        return f"Follow-up search for '{follow_up_query}' found {len(final_results)} refined results:\\n\\n" + "\\n---\\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in follow-up search: {e}")
        return f"Error in follow-up search: {e}"

@mcp.tool()
async def boolean_search(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.001
) -> str:
    """Search with boolean operators (AND, OR, NOT)
    
    Args:
        query: Search query with boolean operators (e.g., 'api AND documentation NOT deprecated')
        limit: Maximum number of results (default: 5)
        min_similarity: Minimum similarity score (default: 0.001)
    
    Returns:
        Formatted search results applying boolean logic
    """
    await init_clients()
    
    try:
        # Parse boolean query
        query_upper = query.upper()
        
        # Split by boolean operators
        if ' AND ' in query_upper:
            parts = re.split(r'\\s+AND\\s+', query, flags=re.IGNORECASE)
            operation = 'AND'
        elif ' OR ' in query_upper:
            parts = re.split(r'\\s+OR\\s+', query, flags=re.IGNORECASE)
            operation = 'OR'
        else:
            # No boolean operators, treat as regular search
            parts = [query]
            operation = 'SINGLE'
        
        # Handle NOT operators
        must_include = []
        must_exclude = []
        
        for part in parts:
            part = part.strip()
            if part.upper().startswith('NOT '):
                must_exclude.append(part[4:].strip())
            else:
                must_include.append(part)
        
        # Get results for main query terms
        all_results = []
        
        for term in must_include:
            term_embedding = await get_embeddings(term)
            term_results = collection.query(
                query_embeddings=[term_embedding],
                n_results=limit * 5,
                include=["documents", "metadatas", "distances"]
            )
            
            for doc, metadata, distance in zip(
                term_results['documents'][0],
                term_results['metadatas'][0],
                term_results['distances'][0]
            ):
                similarity = max(0, 1 - distance)
                if similarity >= min_similarity:
                    all_results.append((doc, metadata, distance, similarity, term))
        
        if not all_results:
            return "No documents found matching boolean query."
        
        # Apply boolean logic
        if operation == 'AND' and len(must_include) > 1:
            # For AND, find documents that match multiple terms
            doc_counts = {}
            for doc, metadata, distance, similarity, term in all_results:
                doc_id = metadata.get('source', '') + str(metadata.get('chunk_id', ''))
                if doc_id not in doc_counts:
                    doc_counts[doc_id] = {'count': 0, 'data': (doc, metadata, distance, similarity)}
                doc_counts[doc_id]['count'] += 1
            
            # Keep only documents that match all terms
            filtered_results = []
            for doc_id, data in doc_counts.items():
                if data['count'] >= len(must_include):
                    filtered_results.append(data['data'])
        else:
            # For OR or single term, use all results
            filtered_results = [(doc, metadata, distance, similarity) for doc, metadata, distance, similarity, term in all_results]
        
        # Apply NOT filters
        if must_exclude:
            final_results = []
            for doc, metadata, distance, similarity in filtered_results:
                exclude_doc = False
                for exclude_term in must_exclude:
                    if exclude_term.lower() in doc.lower():
                        exclude_doc = True
                        break
                if not exclude_doc:
                    final_results.append((doc, metadata, distance, similarity))
        else:
            final_results = filtered_results
        
        # Remove duplicates and sort by similarity
        unique_results = {}
        for doc, metadata, distance, similarity in final_results:
            doc_id = metadata.get('source', '') + str(metadata.get('chunk_id', ''))
            if doc_id not in unique_results or similarity > unique_results[doc_id][3]:
                unique_results[doc_id] = (doc, metadata, distance, similarity)
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x[3], reverse=True)[:limit]
        
        if not sorted_results:
            return f"No documents found matching boolean query: {query}"
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance, similarity) in enumerate(sorted_results):
            formatted_results.append(
                f"**Boolean Result {i+1}** (similarity: {similarity:.3f})\\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')}\\n"
                f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\\n"
            )
        
        return f"Boolean search '{query}' found {len(sorted_results)} documents:\\n\\n" + "\\n---\\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in boolean search: {e}")
        return f"Error in boolean search: {e}"

@mcp.tool()
async def get_search_history(limit: int = 10) -> str:
    """Get recent search history
    
    Args:
        limit: Number of recent searches to return (default: 10)
    
    Returns:
        Formatted list of recent search queries
    """
    try:
        if not search_history:
            return "No search history available."
        
        recent_searches = search_history[:limit]
        
        formatted_history = []
        for i, entry in enumerate(recent_searches, 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M')
            formatted_history.append(
                f"{i}. **{entry['query']}** ({entry['result_count']} results) - {timestamp}"
            )
        
        return f"Recent search history ({len(recent_searches)} searches):\n\n" + "\n".join(formatted_history)
        
    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        return f"Error getting search history: {e}"

@mcp.tool()
async def get_search_suggestions(
    partial_query: str = "",
    limit: int = 10
) -> str:
    """Get search suggestions based on document content and history
    
    Args:
        partial_query: Partial search query to get suggestions for
        limit: Number of suggestions to return (default: 10)
    
    Returns:
        List of suggested search terms
    """
    await init_clients()
    
    try:
        suggestions = []
        
        # Get suggestions from search history
        if search_history:
            history_suggestions = []
            for entry in search_history[:20]:  # Check recent searches
                query = entry['query']
                if not partial_query or partial_query.lower() in query.lower():
                    history_suggestions.append(f"{query} (from history)")
            suggestions.extend(history_suggestions[:limit//2])
        
        # Get suggestions from document keywords
        try:
            keywords = extract_keywords_from_documents()
            keyword_suggestions = []
            
            if partial_query:
                # Filter keywords that match partial query
                matching_keywords = [kw for kw in keywords 
                                   if partial_query.lower() in kw.lower()]
                keyword_suggestions.extend(matching_keywords[:limit//2])
            else:
                # Return popular keywords
                keyword_suggestions.extend(keywords[:limit//2])
            
            suggestions.extend([f"{kw} (keyword)" for kw in keyword_suggestions])
        except:
            pass  # Continue without keywords if extraction fails
        
        # Add common technical search patterns
        if partial_query:
            common_patterns = [
                f"{partial_query} configuration",
                f"{partial_query} setup",
                f"{partial_query} documentation",
                f"{partial_query} tutorial",
                f"{partial_query} error",
                f"{partial_query} example"
            ]
            suggestions.extend([f"{pattern} (pattern)" for pattern in common_patterns[:3]])
        
        if not suggestions:
            return "No suggestions available. Try searching for: configuration, setup, documentation, api, error, tutorial"
        
        # Remove duplicates and limit
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            base_suggestion = suggestion.split(' (')[0]  # Remove type annotation for comparison
            if base_suggestion not in seen:
                seen.add(base_suggestion)
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= limit:
                    break
        
        query_text = f" for '{partial_query}'" if partial_query else ""
        return f"Search suggestions{query_text}:\n\n" + "\n".join([f"• {s}" for s in unique_suggestions])
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return f"Error getting suggestions: {e}"

@mcp.tool()
async def search_with_highlights(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.001,
    snippet_length: int = 300
) -> str:
    """Enhanced search with detailed highlighting and snippets
    
    Args:
        query: The search query
        limit: Maximum number of results (default: 5)
        min_similarity: Minimum similarity score (default: 0.001)
        snippet_length: Length of content snippets (default: 300)
    
    Returns:
        Formatted search results with enhanced highlighting
    """
    await init_clients()
    
    try:
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Search in Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Format results with enhanced highlighting
        formatted_results = []
        query_terms = query.split()
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # Convert cosine distance to similarity
            similarity = max(0, 1 - distance)
            if similarity >= min_similarity:
                # Enhanced highlighting with custom snippet length
                highlighted_content = highlight_text(doc, query_terms, snippet_length)
                
                # Additional metadata
                processed_at = metadata.get('processed_at', 0)
                processed_date = datetime.fromtimestamp(processed_at).strftime('%Y-%m-%d %H:%M') if processed_at else 'Unknown'
                file_type = metadata.get('file_name', '').split('.')[-1].upper() if '.' in metadata.get('file_name', '') else 'Unknown'
                chunk_id = metadata.get('chunk_id', 'N/A')
                
                # Count matching terms
                matching_terms = [term for term in query_terms if term.lower() in doc.lower()]
                
                formatted_results.append(
                    f"**Result {i+1}** (similarity: {similarity:.3f}) [{len(matching_terms)}/{len(query_terms)} terms matched]\n"
                    f"**Source:** {metadata.get('file_name', 'Unknown')} ({file_type}) - Chunk {chunk_id}\n"
                    f"**Processed:** {processed_date}\n"
                    f"**Highlighted Content:**\n{highlighted_content}\n"
                )
        
        if not formatted_results:
            return f"No documents found with similarity >= {min_similarity}"
        
        # Add to search history
        add_to_search_history(query, len(formatted_results))
        
        return f"Enhanced search for '{query}' found {len(formatted_results)} documents:\n\n" + "\n---\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return f"Error in enhanced search: {e}"

@mcp.tool()
async def clear_search_history() -> str:
    """Clear search history
    
    Returns:
        Confirmation message
    """
    global search_history
    count = len(search_history)
    search_history.clear()
    return f"Cleared {count} search history entries."

@mcp.tool()
async def simple_search(query: str, limit: int = 2) -> str:
    """Simple search optimized for local LLMs with concise responses
    
    Args:
        query: The search query
        limit: Maximum results (default: 2, max: 3)
    
    Returns:
        Brief search results in simple format
    """
    await init_clients()
    
    # Limit results for local LLMs
    limit = min(limit, 3)
    
    try:
        query_embedding = await get_embeddings(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No results found."
        
        # Format simple results
        output = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = max(0, 1 - distance)
            # Keep snippets very short for local LLMs
            snippet = doc[:120] + "..." if len(doc) > 120 else doc
            file_name = metadata.get('file_name', 'Unknown')
            
            output.append(f"{i+1}. {file_name}: {snippet}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    logger.info("Starting Local RAG MCP Server with FastMCP")
    logger.info(f"Configuration:")
    logger.info(f"  CHROMA_HOST: {CHROMA_HOST}")
    logger.info(f"  OLLAMA_HOST: {OLLAMA_HOST}")
    logger.info(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    
    # Run the server using stdio transport
    mcp.run(transport='stdio')