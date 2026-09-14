"""Efficient location search service using inverted index."""

import json
import os
from typing import List, Dict, Any, Optional


class LocationSearchService:
    """Provides fast location search with fuzzy matching using inverted index."""
    
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self._load_catalog()
    
    def _load_catalog(self):
        """Load hotel catalog and build inverted index."""
        with open(self.catalog_path, encoding='utf-8') as f:
            self.catalog = json.load(f)
        
        # Build inverted index for efficient search
        self._build_inverted_index()
    
    def _build_inverted_index(self):
        """Build an inverted index for fast location search."""
        self.inverted_index: Dict[str, List[str]] = {}
        self.locations: List[str] = self.catalog.get("places", [])
        
        for location in self.locations:
            # Tokenize location for indexing
            tokens = self._tokenize(location)
            
            # Add each token to inverted index
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(location)
    
    def _tokenize(self, location: str) -> List[str]:
        """Tokenize location string into searchable terms."""
        # Remove parentheses and split by common separators
        clean = location.replace('(', ' ').replace(')', ' ')
        tokens = []
        
        # Add full location
        tokens.append(location.lower())
        
        # Add city name (without state)
        parts = clean.split(',')
        if parts:
            city = parts[0].strip()
            tokens.append(city.lower())
            tokens.extend(city.lower().split())
        
        # Add state abbreviation if present
        state_match = location[location.find('(')+1:location.find(')')] if '(' in location else ''
        if state_match:
            tokens.append(state_match.lower())
        
        return tokens
    
    def search(self, query: str, top_n: int = 5) -> List[str]:
        """Search for locations matching the query with fuzzy matching."""
        if not query or len(query) < 2:
            return []
        
        query_lower = query.lower()
        scored_locations: Dict[str, float] = {}
        
        # Method 1: Direct substring match
        for location in self.locations:
            location_lower = location.lower()
            if query_lower in location_lower:
                # Exact substring match - high score
                score = 100 - len(location_lower)
                scored_locations[location] = score
            elif location_lower in query_lower:
                # Location is substring of query - medium score
                score = 50
                scored_locations[location] = score
        
        # Method 2: Token-based matching (for typos)
        if not scored_locations:
            query_tokens = query_lower.split()
            
            for location in self.locations:
                location_lower = location.lower()
                tokens = self._tokenize(location)
                
                # Count matching tokens
                matches = sum(1 for token in tokens if any(qt in token for qt in query_tokens))
                
                if matches > 0:
                    # Score based on how many tokens match
                    score = matches * 20
                    # Bonus for exact city match
                    city_part = location.split(',')[0].strip() if ',' in location else location
                    if query_lower in city_part.lower():
                        score += 30
                    scored_locations[location] = score
        
        # Method 3: Fuzzy matching with edit distance for typos
        if not scored_locations:
            for location in self.locations:
                # Extract city name from location like "Brasilia (DF)"
                location_clean = location.replace('(', ' ').replace(')', '')
                query_clean = query_lower.replace('(', ' ').replace(')', '')
                
                # Get city part (before any comma or parentheses)
                query_city = query_clean.split(',')[0].split()[0] if query_clean else query_clean
                loc_city = location_clean.split(',')[0].split()[0] if location_clean else location_clean
                
                # Calculate edit distance for city names only
                distance = self._edit_distance(query_city, loc_city)
                
                # Allow matches with distance <= 2 (handles common typos)
                if distance > 0 and distance <= 2:
                    score = 30 - distance * 10  # Higher score for smaller distance
                    scored_locations[location] = score
        
        # Sort by score and return top N
        sorted_locations = sorted(scored_locations.items(), key=lambda x: -x[1])
        return [loc for loc, _ in sorted_locations[:top_n]]
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _fuzzy_match(self, query: str, location: str) -> bool:
        """Check if query matches location with single character difference."""
        # If lengths differ by 1, check for single insertion/deletion
        if abs(len(query) - len(location)) <= 1:
            if len(query) <= len(location):
                shorter, longer = query, location
            else:
                shorter, longer = location, query
            
            # Check if adding one char to shorter makes it equal to longer
            for i in range(len(shorter) + 1):
                # Try inserting a char
                test = shorter[:i] + longer[i] + shorter[i:] if i < len(longer) else shorter
                if test == longer:
                    return True
            
            # Try single character replacement
            if len(query) == len(location):
                differences = sum(1 for a, b in zip(query, location) if a != b)
                if differences == 1:
                    return True
        
        return False
    
    def get_all_locations(self) -> List[str]:
        """Get all available locations."""
        return self.locations.copy()


# Global instance
_location_search_service: Optional[LocationSearchService] = None


def get_location_search_service() -> LocationSearchService:
    """Get or create the location search service instance."""
    global _location_search_service
    
    if _location_search_service is None:
        from config.settings import settings
        _location_search_service = LocationSearchService(settings.hotels_catalog_path)
    
    return _location_search_service


def reset_location_search_service():
    """Reset the location search service (for testing)."""
    global _location_search_service
    _location_search_service = None
