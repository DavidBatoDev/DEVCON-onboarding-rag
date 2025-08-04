# Source Deduplication Implementation

## Overview

This document describes the implementation of source deduplication in the DEVCON chatbot backend to prevent duplicate sources from appearing in responses.

## Problem

Previously, the backend could return duplicate sources in the "📚 **Sources Referenced:**" section when:

1. Multiple chunks from the same document were retrieved
2. The same document was referenced multiple times in the response
3. The `query_with_reference_control` method was called, which could add sources twice

## Solution

### 1. Deduplication Helper Method

Added a new helper method `_deduplicate_sources()` in `LlamaIndexRAGService`:

```python
def _deduplicate_sources(self, source_nodes, max_sources: int = 3):
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
```

### 2. Updated Query Methods

Both `query_with_history_and_verification()` and `query_with_reference_control()` now use the deduplication helper:

```python
# Before
for i, node in enumerate(response.source_nodes[:3], 1):
    # ... add source

# After
unique_sources = self._deduplicate_sources(response.source_nodes, max_sources=3)
for i, node in enumerate(unique_sources, 1):
    # ... add source
```

### 3. Fixed Reference Control Method

The `query_with_reference_control()` method was refactored to:

- Avoid calling `query_with_history_and_verification()` when `force_show_references` is specified
- Directly control the query process to prevent double source addition
- Use the same deduplication logic

## Key Features

### Deduplication Logic

- **Unique Identifier**: Combines `title` and `file_id` to create a unique key
- **Order Preservation**: Maintains the original order of sources
- **Limit Control**: Respects the `max_sources` parameter (default: 3)
- **Fallback Handling**: Uses `source` field if `title` is not available

### Edge Case Handling

- **Empty Lists**: Returns empty list when no sources provided
- **Single Sources**: Handles single source correctly
- **All Identical**: Deduplicates when all sources are identical
- **Missing Metadata**: Gracefully handles missing metadata fields

## Testing

A comprehensive test suite was created in `scripts/test_deduplication_simple.py` that verifies:

1. **Basic Deduplication**: Removes duplicate sources correctly
2. **Order Preservation**: Maintains original order of unique sources
3. **Limit Enforcement**: Respects maximum source limit
4. **Edge Cases**: Handles empty lists, single sources, and identical sources

### Test Results

```
🧪 Testing source deduplication logic...
📊 Original nodes: 6
📊 After deduplication: 3
✅ Deduplication successful!

🧪 Testing edge cases...
✅ Empty list handled correctly
✅ Single node handled correctly
✅ Identical nodes handled correctly

🎉 All deduplication tests passed!
```

## Benefits

1. **Cleaner Responses**: No more duplicate sources in responses
2. **Better UX**: Users see unique, relevant sources only
3. **Consistent Behavior**: Both query methods now behave consistently
4. **Maintainable Code**: Centralized deduplication logic
5. **Performance**: Efficient O(n) deduplication with early termination

## Implementation Details

### Files Modified

- `server/app/services/llamaindex_service.py`
  - Added `_deduplicate_sources()` helper method
  - Updated `query_with_history_and_verification()` method
  - Refactored `query_with_reference_control()` method

### Files Added

- `server/scripts/test_deduplication_simple.py` - Test suite
- `server/SOURCE_DEDUPLICATION.md` - This documentation

## Future Enhancements

1. **Configurable Limits**: Allow runtime configuration of max sources
2. **Source Ranking**: Prioritize sources by relevance score
3. **Metadata Enrichment**: Add more metadata fields for better deduplication
4. **Caching**: Cache deduplication results for performance
