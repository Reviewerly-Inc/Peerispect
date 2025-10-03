#!/usr/bin/env python3
"""
Test script for API with positional chunking and highlighting data
"""

import requests
import json
import time

def test_api_positional():
    """Test the API with positional chunking enabled"""
    
    # API endpoint
    base_url = "http://localhost:5015"
    
    # Test URL (using the same paper we processed before)
    test_url = "https://openreview.net/forum?id=odjMSBSWRt"
    
    print("🚀 Testing API with positional chunking...")
    print(f"📄 Processing: {test_url}")
    print("=" * 60)
    
    # Make request
    payload = {
        "openreview_url": test_url,
        "config_overrides": {
            "positional_chunking": True,
            "column_split_x": 300.0
        }
    }
    
    try:
        response = requests.post(f"{base_url}/process", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received!")
            print(f"📊 Status: {result['status']}")
            print(f"⏱️  Processing time: {result.get('processing_time', 'N/A')} seconds")
            print(f"📄 PDF URL: {result.get('pdf_url', 'N/A')}")
            
            # Check positional data
            positional_data = result.get('positional_data', {})
            if positional_data:
                chunks = positional_data.get('chunks', {})
                print(f"🎯 Positional chunks loaded: {len(chunks)}")
                
                # Show sample chunk
                if chunks:
                    sample_chunk_id = list(chunks.keys())[0]
                    sample_chunk = chunks[sample_chunk_id]
                    print(f"📋 Sample chunk: {sample_chunk_id}")
                    print(f"   Pages: {sample_chunk['pages']}")
                    print(f"   Positions: {len(sample_chunk['positions'])}")
                    print(f"   Text: {sample_chunk['text'][:100]}...")
            
            # Check highlighting map
            highlighting_map = result.get('highlighting_map', {})
            if highlighting_map:
                print(f"🎨 Highlighting map created for {len(highlighting_map)} claims")
                
                # Show sample highlighting
                if highlighting_map:
                    sample_claim_id = list(highlighting_map.keys())[0]
                    sample_highlighting = highlighting_map[sample_claim_id]
                    print(f"📌 Sample claim highlighting: {sample_claim_id}")
                    print(f"   Evidence highlights: {len(sample_highlighting['evidence_highlights'])}")
            
            # Check verification results
            results = result.get('results', [])
            if results:
                print(f"🔍 Verification results: {len(results)} claims")
                
                # Show sample result
                if results:
                    sample_result = results[0]
                    print(f"📝 Sample claim: {sample_result.get('claim', '')[:100]}...")
                    print(f"   Evidence count: {len(sample_result.get('evidence', []))}")
                    print(f"   Verification: {sample_result.get('verification_result', 'N/A')}")
            
            print("\n🎉 API test completed successfully!")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:5015")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_positional()
