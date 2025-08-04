#!/usr/bin/env python3
"""
Simple test script to verify source deduplication logic
"""

class MockNode:
    """Mock node class for testing"""
    def __init__(self, title, source, file_id):
        self.metadata = {
            'title': title,
            'source': source,
            'file_id': file_id
        }

def deduplicate_sources(source_nodes, max_sources=3):
    """Deduplicate source nodes based on title and file_id"""
    seen_sources = set()
    unique_sources = []
    
    for node in source_nodes:
        source_info = node.metadata.get('title', node.metadata.get('source', 'Unknown'))
        file_id = node.metadata.get('file_id')
        
        # Create a unique identifier for this source
        source_key = f"{source_info}_{file_id}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            unique_sources.append(node)
        
        # Limit to max_sources
        if len(unique_sources) >= max_sources:
            break
    
    return unique_sources

def test_deduplication():
    """Test the deduplication logic"""
    print("🧪 Testing source deduplication logic...")
    
    # Create test nodes with duplicates
    test_nodes = [
        MockNode("Test Document 1", "test1.pdf", "12345"),
        MockNode("Test Document 1", "test1.pdf", "12345"),  # Duplicate
        MockNode("Test Document 2", "test2.pdf", "67890"),
        MockNode("Test Document 1", "test1.pdf", "12345"),  # Another duplicate
        MockNode("Test Document 3", "test3.pdf", "11111"),
        MockNode("Test Document 2", "test2.pdf", "67890"),  # Duplicate of doc 2
    ]
    
    print(f"📊 Original nodes: {len(test_nodes)}")
    for i, node in enumerate(test_nodes, 1):
        print(f"  {i}. {node.metadata['title']} ({node.metadata['file_id']})")
    
    # Apply deduplication
    unique_nodes = deduplicate_sources(test_nodes, max_sources=3)
    
    print(f"\n📊 After deduplication: {len(unique_nodes)}")
    for i, node in enumerate(unique_nodes, 1):
        print(f"  {i}. {node.metadata['title']} ({node.metadata['file_id']})")
    
    # Verify results
    expected_unique = 3  # Should have 3 unique sources
    if len(unique_nodes) == expected_unique:
        print(f"\n✅ Deduplication successful! Expected {expected_unique}, got {len(unique_nodes)}")
        
        # Check that we have the expected unique sources
        unique_titles = set()
        for node in unique_nodes:
            source_key = f"{node.metadata['title']}_{node.metadata['file_id']}"
            unique_titles.add(source_key)
        
        print(f"📊 Unique source keys: {len(unique_titles)}")
        for key in unique_titles:
            print(f"  - {key}")
        
        return True
    else:
        print(f"\n❌ Deduplication failed! Expected {expected_unique}, got {len(unique_nodes)}")
        return False

def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing edge cases...")
    
    # Test with no nodes
    empty_result = deduplicate_sources([])
    if len(empty_result) == 0:
        print("✅ Empty list handled correctly")
    else:
        print("❌ Empty list not handled correctly")
        return False
    
    # Test with single node
    single_node = [MockNode("Single Doc", "single.pdf", "99999")]
    single_result = deduplicate_sources(single_node)
    if len(single_result) == 1:
        print("✅ Single node handled correctly")
    else:
        print("❌ Single node not handled correctly")
        return False
    
    # Test with all identical nodes
    identical_nodes = [
        MockNode("Same Doc", "same.pdf", "11111"),
        MockNode("Same Doc", "same.pdf", "11111"),
        MockNode("Same Doc", "same.pdf", "11111"),
    ]
    identical_result = deduplicate_sources(identical_nodes)
    if len(identical_result) == 1:
        print("✅ Identical nodes handled correctly")
    else:
        print("❌ Identical nodes not handled correctly")
        return False
    
    return True

if __name__ == "__main__":
    success1 = test_deduplication()
    success2 = test_edge_cases()
    
    if success1 and success2:
        print("\n🎉 All deduplication tests passed!")
    else:
        print("\n💥 Some deduplication tests failed!")
        exit(1) 