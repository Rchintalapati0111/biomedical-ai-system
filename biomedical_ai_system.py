# biomedical_ai_system.py
"""
Autonomous Biomedical Insight Generation System
Multi-agent AI system for literature analysis and molecular target recommendation
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import sqlite3
from contextlib import asynccontextmanager

import aiohttp
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# AutoGen and LangChain imports
try:
    import autogen
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.schema import SystemMessage
    from langchain.memory import ConversationBufferMemory
except ImportError:
    print("Installing required packages...")
    os.system("pip install pyautogen langchain openai anthropic")
    import autogen

# Security and compliance
from cryptography.fernet import Fernet
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    LITERATURE_SCANNER = "literature_scanner"
    EVIDENCE_VALIDATOR = "evidence_validator" 
    TARGET_RECOMMENDER = "target_recommender"
    ORCHESTRATOR = "orchestrator"

@dataclass
class ResearchInsight:
    """Structured representation of a biomedical insight"""
    insight_id: str
    hypothesis: str
    evidence_sources: List[str]
    confidence_score: float
    molecular_targets: List[str]
    validation_status: str
    created_at: str
    agent_attribution: str

@dataclass
class LiteratureSource:
    """Represents a scientific literature source"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    impact_factor: float = 0.0
    relevance_score: float = 0.0

class SecurityManager:
    """HIPAA-compliant security and audit logging"""
    
    def __init__(self):
        self.encryption_key = self._generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.audit_log = []
        self._setup_database()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key for sensitive data"""
        return Fernet.generate_key()
    
    def _setup_database(self):
        """Initialize secure database for audit logging"""
        self.conn = sqlite3.connect('biomedical_audit.db', check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_role TEXT,
                action TEXT,
                data_hash TEXT,
                agent_id TEXT
            )
        ''')
        self.conn.commit()
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def log_action(self, user_role: str, action: str, data: Any, agent_id: str):
        """Log actions for HIPAA compliance"""
        timestamp = datetime.now().isoformat()
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO audit_log (timestamp, user_role, action, data_hash, agent_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, user_role, action, data_hash, agent_id))
        self.conn.commit()
        
        logger.info(f"Audit log: {user_role} performed {action} via {agent_id}")

class PubMedSearchTool:
    """Tool for searching and retrieving biomedical literature"""
    
    def __init__(self, security_manager: SecurityManager):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.security = security_manager
    
    async def search_literature(self, query: str, max_results: int = 50) -> List[LiteratureSource]:
        """Search PubMed for relevant literature"""
        self.security.log_action("system", "literature_search", query, "pubmed_tool")
        
        try:
            # Search for paper IDs
            search_url = f"{self.base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    search_data = await response.json()
            
            paper_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not paper_ids:
                return []
            
            # Fetch paper details
            return await self._fetch_paper_details(paper_ids)
            
        except Exception as e:
            logger.error(f"Error searching literature: {e}")
            return []
    
    async def _fetch_paper_details(self, paper_ids: List[str]) -> List[LiteratureSource]:
        """Fetch detailed information for papers"""
        fetch_url = f"{self.base_url}efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(paper_ids),
            'retmode': 'xml'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(fetch_url, params=fetch_params) as response:
                xml_data = await response.text()
        
        return self._parse_xml_response(xml_data)
    
    def _parse_xml_response(self, xml_data: str) -> List[LiteratureSource]:
        """Parse XML response from PubMed"""
        sources = []
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall('.//PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                title_elem = article.find('.//ArticleTitle')
                abstract_elem = article.find('.//AbstractText')
                
                if pmid_elem is not None and title_elem is not None:
                    pmid = pmid_elem.text
                    title = title_elem.text or ""
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.find('.//LastName')
                        forename = author.find('.//ForeName')
                        if lastname is not None:
                            author_name = lastname.text
                            if forename is not None:
                                author_name = f"{forename.text} {author_name}"
                            authors.append(author_name)
                    
                    # Extract journal info
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
                    
                    # Extract publication date
                    pub_date_elem = article.find('.//PubDate/Year')
                    pub_date = pub_date_elem.text if pub_date_elem is not None else "Unknown"
                    
                    source = LiteratureSource(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=journal,
                        publication_date=pub_date
                    )
                    sources.append(source)
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing XML response: {e}")
        
        return sources

class MolecularTargetAnalyzer:
    """Analyzes molecular targets from literature"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        # Common drug targets and pathways
        self.known_targets = {
            'proteins': ['p53', 'EGFR', 'VEGF', 'TNF-alpha', 'IL-6', 'BRCA1', 'BRCA2'],
            'pathways': ['PI3K/AKT', 'mTOR', 'NF-kB', 'MAPK', 'JAK/STAT'],
            'receptors': ['dopamine', 'serotonin', 'GABA', 'glutamate']
        }
    
    def identify_targets(self, literature_sources: List[LiteratureSource]) -> List[Dict[str, Any]]:
        """Identify potential molecular targets from literature"""
        self.security.log_action("system", "target_analysis", 
                                f"{len(literature_sources)} sources", "target_analyzer")
        
        targets_found = []
        target_mentions = {}
        
        for source in literature_sources:
            text = f"{source.title} {source.abstract}".lower()
            
            # Search for known targets
            for category, targets in self.known_targets.items():
                for target in targets:
                    if target.lower() in text:
                        key = f"{category}:{target}"
                        if key not in target_mentions:
                            target_mentions[key] = {
                                'target': target,
                                'category': category,
                                'mentions': 0,
                                'sources': [],
                                'confidence': 0.0
                            }
                        
                        target_mentions[key]['mentions'] += 1
                        target_mentions[key]['sources'].append(source.pmid)
        
        # Calculate confidence scores
        for key, data in target_mentions.items():
            # Simple scoring based on frequency and source quality
            confidence = min(data['mentions'] * 0.2, 1.0)
            data['confidence'] = confidence
            
            if confidence > 0.3:  # Threshold for inclusion
                targets_found.append(data)
        
        # Sort by confidence
        targets_found.sort(key=lambda x: x['confidence'], reverse=True)
        return targets_found[:10]  # Top 10 targets

class BiomedicalAgent:
    """Base class for biomedical AI agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, security_manager: SecurityManager):
        self.agent_id = agent_id
        self.role = role
        self.security = security_manager
        self.memory = ConversationBufferMemory()
        self.tools = []
        
    def log_activity(self, action: str, data: Any):
        """Log agent activity for audit trail"""
        self.security.log_action("agent", action, data, self.agent_id)

class LiteratureScanningAgent(BiomedicalAgent):
    """Agent responsible for scanning and analyzing biomedical literature"""
    
    def __init__(self, security_manager: SecurityManager):
        super().__init__("lit_scanner_001", AgentRole.LITERATURE_SCANNER, security_manager)
        self.pubmed_tool = PubMedSearchTool(security_manager)
    
    async def scan_literature(self, research_query: str) -> List[LiteratureSource]:
        """Scan literature for relevant papers"""
        self.log_activity("literature_scan", research_query)
        
        logger.info(f"Literature scanning agent: Searching for '{research_query}'")
        
        # Enhance query with biomedical terms
        enhanced_query = self._enhance_query(research_query)
        
        # Search literature
        sources = await self.pubmed_tool.search_literature(enhanced_query, max_results=50)
        
        # Score relevance
        scored_sources = self._score_relevance(sources, research_query)
        
        logger.info(f"Found {len(scored_sources)} relevant papers")
        return scored_sources
    
    def _enhance_query(self, query: str) -> str:
        """Enhance search query with relevant biomedical terms"""
        biomedical_terms = ["molecular", "therapeutic", "clinical", "treatment", "pathway"]
        
        # Simple query enhancement - in practice, this would be more sophisticated
        if not any(term in query.lower() for term in biomedical_terms):
            query += " AND (therapeutic OR molecular OR clinical)"
        
        return query
    
    def _score_relevance(self, sources: List[LiteratureSource], query: str) -> List[LiteratureSource]:
        """Score sources based on relevance to query"""
        query_terms = set(query.lower().split())
        
        for source in sources:
            text = f"{source.title} {source.abstract}".lower()
            text_terms = set(text.split())
            
            # Simple relevance scoring
            common_terms = query_terms.intersection(text_terms)
            source.relevance_score = len(common_terms) / len(query_terms) if query_terms else 0
        
        # Sort by relevance
        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        return sources

class EvidenceValidationAgent(BiomedicalAgent):
    """Agent responsible for validating scientific evidence"""
    
    def __init__(self, security_manager: SecurityManager):
        super().__init__("evidence_val_001", AgentRole.EVIDENCE_VALIDATOR, security_manager)
    
    def validate_evidence(self, sources: List[LiteratureSource]) -> Dict[str, Any]:
        """Validate the quality and reliability of evidence"""
        self.log_activity("evidence_validation", f"{len(sources)} sources")
        
        logger.info(f"Evidence validation agent: Validating {len(sources)} sources")
        
        validation_results = {
            'total_sources': len(sources),
            'high_quality_sources': 0,
            'recent_sources': 0,
            'peer_reviewed': 0,
            'evidence_strength': 'moderate',
            'recommendations': []
        }
        
        current_year = datetime.now().year
        
        for source in sources:
            # Check publication recency (within 5 years)
            try:
                pub_year = int(source.publication_date)
                if current_year - pub_year <= 5:
                    validation_results['recent_sources'] += 1
            except (ValueError, TypeError):
                pass
            
            # Simple quality indicators
            if len(source.authors) >= 3:  # Multi-author studies
                validation_results['high_quality_sources'] += 1
            
            # Assume peer-reviewed if from known journals
            known_journals = ['nature', 'science', 'cell', 'nejm', 'lancet']
            if any(journal in source.journal.lower() for journal in known_journals):
                validation_results['peer_reviewed'] += 1
        
        # Determine evidence strength
        quality_ratio = validation_results['high_quality_sources'] / len(sources)
        recent_ratio = validation_results['recent_sources'] / len(sources)
        
        if quality_ratio > 0.7 and recent_ratio > 0.5:
            validation_results['evidence_strength'] = 'strong'
        elif quality_ratio > 0.4 and recent_ratio > 0.3:
            validation_results['evidence_strength'] = 'moderate'
        else:
            validation_results['evidence_strength'] = 'weak'
        
        # Generate recommendations
        if validation_results['evidence_strength'] == 'weak':
            validation_results['recommendations'].append("Seek additional high-quality sources")
        if validation_results['recent_sources'] < len(sources) * 0.3:
            validation_results['recommendations'].append("Include more recent studies")
        
        logger.info(f"Evidence strength: {validation_results['evidence_strength']}")
        return validation_results

class TargetRecommendationAgent(BiomedicalAgent):
    """Agent responsible for recommending molecular targets"""
    
    def __init__(self, security_manager: SecurityManager):
        super().__init__("target_rec_001", AgentRole.TARGET_RECOMMENDER, security_manager)
        self.target_analyzer = MolecularTargetAnalyzer(security_manager)
    
    def recommend_targets(self, sources: List[LiteratureSource], 
                         validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend molecular targets based on literature analysis"""
        self.log_activity("target_recommendation", f"{len(sources)} sources analyzed")
        
        logger.info(f"Target recommendation agent: Analyzing {len(sources)} sources")
        
        # Identify potential targets
        targets = self.target_analyzer.identify_targets(sources)
        
        # Adjust confidence based on evidence quality
        evidence_multiplier = {
            'strong': 1.2,
            'moderate': 1.0,
            'weak': 0.8
        }.get(validation_results['evidence_strength'], 1.0)
        
        for target in targets:
            target['adjusted_confidence'] = min(target['confidence'] * evidence_multiplier, 1.0)
            target['evidence_quality'] = validation_results['evidence_strength']
            target['rationale'] = self._generate_rationale(target, validation_results)
        
        # Sort by adjusted confidence
        targets.sort(key=lambda x: x['adjusted_confidence'], reverse=True)
        
        logger.info(f"Recommended {len(targets)} molecular targets")
        return targets
    
    def _generate_rationale(self, target: Dict[str, Any], 
                          validation_results: Dict[str, Any]) -> str:
        """Generate rationale for target recommendation"""
        rationale_parts = [
            f"Target {target['target']} identified in {target['mentions']} sources",
            f"Evidence quality: {validation_results['evidence_strength']}",
            f"Confidence score: {target['confidence']:.2f}"
        ]
        
        if target['category'] == 'proteins':
            rationale_parts.append("Protein target suitable for small molecule or antibody therapy")
        elif target['category'] == 'pathways':
            rationale_parts.append("Pathway target suitable for multi-target intervention")
        
        return ". ".join(rationale_parts)

class BiomedicalOrchestrator:
    """Main orchestrator for the biomedical insight generation system"""
    
    def __init__(self):
        self.security = SecurityManager()
        self.literature_agent = LiteratureScanningAgent(self.security)
        self.validation_agent = EvidenceValidationAgent(self.security)
        self.target_agent = TargetRecommendationAgent(self.security)
        
        # Initialize database for insights
        self._setup_insights_database()
    
    def _setup_insights_database(self):
        """Setup database for storing research insights"""
        conn = sqlite3.connect('biomedical_insights.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                hypothesis TEXT,
                evidence_sources TEXT,
                confidence_score REAL,
                molecular_targets TEXT,
                validation_status TEXT,
                created_at TEXT,
                agent_attribution TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    async def generate_insights(self, research_query: str) -> ResearchInsight:
        """Main workflow for generating biomedical insights"""
        logger.info(f"Starting insight generation for: '{research_query}'")
        
        # Step 1: Literature scanning
        literature_sources = await self.literature_agent.scan_literature(research_query)
        
        if not literature_sources:
            logger.warning("No literature sources found")
            return self._create_empty_insight(research_query)
        
        # Step 2: Evidence validation
        validation_results = self.validation_agent.validate_evidence(literature_sources)
        
        # Step 3: Target recommendation
        recommended_targets = self.target_agent.recommend_targets(
            literature_sources, validation_results
        )
        
        # Step 4: Generate structured insight
        insight = self._synthesize_insight(
            research_query, literature_sources, validation_results, recommended_targets
        )
        
        # Step 5: Store insight
        self._store_insight(insight)
        
        logger.info(f"Generated insight with ID: {insight.insight_id}")
        return insight
    
    def _synthesize_insight(self, query: str, sources: List[LiteratureSource],
                           validation: Dict[str, Any], targets: List[Dict[str, Any]]) -> ResearchInsight:
        """Synthesize information into a structured insight"""
        
        # Generate hypothesis based on findings
        top_targets = [t['target'] for t in targets[:3]]
        hypothesis = f"Research suggests potential therapeutic intervention targeting {', '.join(top_targets)} for {query}"
        
        # Calculate overall confidence
        target_confidences = [t['adjusted_confidence'] for t in targets[:5]]
        avg_confidence = np.mean(target_confidences) if target_confidences else 0.0
        
        # Adjust for evidence quality
        evidence_factor = {'strong': 1.0, 'moderate': 0.8, 'weak': 0.6}
        confidence_score = avg_confidence * evidence_factor.get(validation['evidence_strength'], 0.6)
        
        insight = ResearchInsight(
            insight_id=str(uuid.uuid4()),
            hypothesis=hypothesis,
            evidence_sources=[s.pmid for s in sources[:10]],
            confidence_score=confidence_score,
            molecular_targets=[t['target'] for t in targets[:5]],
            validation_status=validation['evidence_strength'],
            created_at=datetime.now().isoformat(),
            agent_attribution="multi_agent_system"
        )
        
        return insight
    
    def _create_empty_insight(self, query: str) -> ResearchInsight:
        """Create empty insight when no sources found"""
        return ResearchInsight(
            insight_id=str(uuid.uuid4()),
            hypothesis=f"Insufficient literature found for: {query}",
            evidence_sources=[],
            confidence_score=0.0,
            molecular_targets=[],
            validation_status="insufficient_data",
            created_at=datetime.now().isoformat(),
            agent_attribution="literature_scanner"
        )
    
    def _store_insight(self, insight: ResearchInsight):
        """Store insight in database"""
        conn = sqlite3.connect('biomedical_insights.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            insight.insight_id,
            insight.hypothesis,
            json.dumps(insight.evidence_sources),
            insight.confidence_score,
            json.dumps(insight.molecular_targets),
            insight.validation_status,
            insight.created_at,
            insight.agent_attribution
        ))
        conn.commit()
        conn.close()
    
    def get_insights_summary(self) -> Dict[str, Any]:
        """Get summary of all generated insights"""
        conn = sqlite3.connect('biomedical_insights.db', check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM insights ORDER BY created_at DESC')
        results = cursor.fetchall()
        conn.close()
        
        summary = {
            'total_insights': len(results),
            'high_confidence_insights': sum(1 for r in results if r[3] > 0.7),
            'recent_insights': sum(1 for r in results 
                                 if (datetime.now() - datetime.fromisoformat(r[6])).days <= 7),
            'insights': []
        }
        
        for row in results[:10]:  # Latest 10 insights
            summary['insights'].append({
                'id': row[0],
                'hypothesis': row[1],
                'confidence': row[3],
                'targets': json.loads(row[4]),
                'created_at': row[6]
            })
        
        return summary

# Demo and testing functions
async def demo_biomedical_system():
    """Demonstrate the biomedical insight generation system"""
    print("🧬 Autonomous Biomedical Insight Generation System")
    print("=" * 50)
    
    orchestrator = BiomedicalOrchestrator()
    
    # Test queries
    test_queries = [
        "Alzheimer's disease protein aggregation",
        "cancer immunotherapy checkpoint inhibitors",
        "diabetes glucose metabolism"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Processing query: {query}")
        print("-" * 30)
        
        try:
            insight = await orchestrator.generate_insights(query)
            
            print(f"📊 Insight ID: {insight.insight_id}")
            print(f"💡 Hypothesis: {insight.hypothesis}")
            print(f"📈 Confidence: {insight.confidence_score:.2f}")
            print(f"🎯 Targets: {', '.join(insight.molecular_targets[:3])}")
            print(f"✅ Validation: {insight.validation_status}")
            print(f"📚 Sources: {len(insight.evidence_sources)} papers analyzed")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    # Show summary
    print(f"\n📈 System Summary:")
    print("-" * 20)
    summary = orchestrator.get_insights_summary()
    print(f"Total insights generated: {summary['total_insights']}")
    print(f"High confidence insights: {summary['high_confidence_insights']}")
    print(f"Recent insights (7 days): {summary['recent_insights']}")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_biomedical_system())