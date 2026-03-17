using Test
using GraphMERT
using GraphMERT: SQLiteUMLSCache, get_cached_item, set_cached_item, create_umls_client, InMemoryUMLSCache
using SQLite
using Dates

@testset "SQLite UMLS Cache" begin
    # Use a temp file for the database
    db_path = tempname() * ".sqlite"
    
    try
        # 1. Test creation and initialization
        cache = SQLiteUMLSCache(db_path, 3600)
        @test isfile(db_path)
        
        # Verify table structure
        # Explicitly close this connection to avoid locking
        # db = SQLite.DB(db_path)
        # tables = SQLite.tables(db)
        # @test "umls_cache" in [t.name for t in tables]
        # DBInterface.close!(db)
        
        # 2. Test Set/Get
        test_data = Dict("foo" => "bar", "val" => 123)
        set_cached_item(cache, "test_cat", "key1", test_data)
        
        retrieved = get_cached_item(cache, "test_cat", "key1")
        @test retrieved !== nothing
        @test retrieved["foo"] == "bar"
        @test retrieved["val"] == 123
        
        # Test missing key
        @test get_cached_item(cache, "test_cat", "missing") === nothing
        
        # Test different category
        @test get_cached_item(cache, "other_cat", "key1") === nothing
        
        # 3. Test Persistence (Close and Reopen)
        # Force db close (SQLite.jl handles this via GC usually, but we can just open a new handle)
        # In test environment, multiple handles to same file might cause locking if not careful
        # Let's rely on SQLite's concurrency or just close cache manually if possible
        # Since SQLiteUMLSCache struct holds db, we can close it
        DBInterface.close!(cache.db)
        
        cache2 = SQLiteUMLSCache(db_path, 3600)
        retrieved2 = get_cached_item(cache2, "test_cat", "key1")
        @test retrieved2 !== nothing
        @test retrieved2["foo"] == "bar"
        DBInterface.close!(cache2.db)
        
        # 4. Test TTL Expiration
        # Create cache with 1 second TTL
        short_cache = SQLiteUMLSCache(db_path, 1)
        set_cached_item(short_cache, "ttl_cat", "key_ttl", "data")
        
        @test get_cached_item(short_cache, "ttl_cat", "key_ttl") == "data"
        
        sleep(1.2) # Wait for expiration
        
        @test get_cached_item(short_cache, "ttl_cat", "key_ttl") === nothing
        DBInterface.close!(short_cache.db)
        
        # 5. Test Client Integration
        client = create_umls_client("mock", cache_path=db_path)
        @test client.cache isa SQLiteUMLSCache
        DBInterface.close!(client.cache.db)
        
    finally
        # Cleanup
        rm(db_path, force=true)
    end
end
