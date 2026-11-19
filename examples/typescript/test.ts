import { VectorDB } from ".";

async function runTests() {
    try {
        // Initialize the vector database
        const vectorDB = new VectorDB();

        // Test 1: Insert some sample documents
        await vectorDB.insert(
            "The quick brown fox jumps over the lazy dog",
        );
        await vectorDB.insert(
            "Machine learning is a subset of artificial intelligence",
        );
        // Test 2: Search for similar documents
        const results = await vectorDB.search("What is artificial intelligence and llm?", 3);
        console.info('Search results:', {
            query: "What is artificial intelligence?",
            results: results
        });

        // Test 3: Search for unrelated content
        console.info('Test 3: Searching for unrelated content...');
        const unrelatedResults = await vectorDB.search("Recipe for chocolate cake", 2);
        console.info('Unrelated search results:', {
            query: "Recipe for chocolate cake",
            results: unrelatedResults
        });

    } catch (error) {
        console.error('Test failed:', error);
    } finally {
        // Clean up
        vectorDB.getDatabase().close();
    }
}

// Run the tests
runTests().then(() => {
    console.log('Tests completed');
}); 