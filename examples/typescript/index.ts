import Database from 'better-sqlite3';
import vectorlite from 'vectorlite';
import { OpenAI } from 'openai';
import { join } from 'path';
import { existsSync, mkdirSync } from 'fs';


export class VectorDB {
  private db!: Database.Database;
  private openai!: OpenAI;
  private nextRowId!: number;
  private readonly indexFilePath: string;
  private readonly dbFilePath: string;
  private maxElements: number;
  private static readonly DEFAULT_INDEX_FILE = 'index_file.bin';
  private static readonly DEFAULT_DB_FILE = 'vector_store.db';
  private static readonly DEFAULT_DATA_DIR = join(process.cwd(), 'data', 'vector-db');

  constructor(
    dataDir?: string,
    indexFilePath: string = VectorDB.DEFAULT_INDEX_FILE,
    dbFilePath: string = VectorDB.DEFAULT_DB_FILE,
    maxElements: number = 100000,
  ) {
    const targetDir = dataDir ? join(process.cwd(), dataDir) : VectorDB.DEFAULT_DATA_DIR;

    if (!existsSync(targetDir)) {
      mkdirSync(targetDir, { recursive: true });
    }

    this.indexFilePath = join(targetDir, indexFilePath);
    this.dbFilePath = join(targetDir, dbFilePath);
    this.nextRowId = 1;
    this.maxElements = maxElements;
    this.openai = new OpenAI({
      apiKey: <OpenAI API Key>,
    });

    this.initializeDatabase();
  }

  private initializeDatabase(): void {
    const extensionPath = vectorlite.vectorlitePath();

    this.db = new Database(this.dbFilePath);
    this.db.loadExtension(extensionPath);

    this.createTables();

    const maxRowIdResult = this.db
      .prepare(
        `
        SELECT MAX(rowid) as maxId FROM content_store
        `,
      )
      .get() as { maxId: number | null };
    this.nextRowId = (maxRowIdResult?.maxId || 0) + 1;
  }

  private createTables(): void {
    this.db.exec(`
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_index USING vectorlite(
                embedding_vector float32[1536], 
                hnsw(max_elements=${this.maxElements}),
                '${this.indexFilePath}'
            );
            
            CREATE TABLE IF NOT EXISTS content_store (
                rowid INTEGER PRIMARY KEY,
                content TEXT
            );
        `);
  }

  private async getEmbedding(text: string): Promise<number[]> {
    try {
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
      });
      return response.data[0].embedding;
    } catch (error) {
      throw error;
    }
  }

  async insert(content: string): Promise<boolean> {
    if (!content || content.trim().length === 0) {
      throw new Error('Content cannot be empty');
    }

    const embedding = await this.getEmbedding(content);

    this.db.exec('BEGIN');
    try {
      if (this.nextRowId > this.maxElements) {
        const oldestRowId = this.db
          .prepare(
            `
                    SELECT MIN(rowid) as min_id FROM content_store
                `,
          )
          .get() as { min_id: number };
        const oldestId = Number(oldestRowId.min_id);

        try {
          this.db.exec(`DELETE FROM embeddings_index WHERE rowid = ${oldestId}`);
          this.db.exec(`DELETE FROM content_store WHERE rowid = ${oldestId}`);
        } catch (deleteError) {
          throw deleteError;
        }

        this.nextRowId = oldestId;
      }

      const currentRowId = Number(this.nextRowId);
      const vectorStmt = this.db.prepare(`
                INSERT INTO embeddings_index (rowid, embedding_vector) 
                VALUES (?, ?)
            `);

      const contentStmt = this.db.prepare(`
                INSERT INTO content_store (rowid, content)
                VALUES (?, ?)
            `);

      vectorStmt.run(currentRowId, Buffer.from(new Float32Array(embedding).buffer));

      contentStmt.run(currentRowId, content);

      this.nextRowId++;
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }

    return true;
  }

  async search(
    content: string,
    limit: number = 5,
  ): Promise<Array<{ rowid: number; distance: number; content: string }>> {
    if (!content || content.trim().length === 0) {
      throw new Error('Search query cannot be empty');
    }

    const embedding = await this.getEmbedding(content);

    const integerLimit = parseInt(limit.toString(), 10);
    const stmt = this.db.prepare(`
            SELECT v.rowid, v.distance, c.content
            FROM (
                SELECT rowid, distance 
                FROM embeddings_index 
                WHERE knn_search(embedding_vector, knn_param(?, ${integerLimit}))
            ) v
            JOIN content_store c ON v.rowid = c.rowid
        `);

    return stmt.all(Buffer.from(new Float32Array(embedding).buffer)) as Array<{
      rowid: number;
      distance: number;
      content: string;
    }>;
  }

  getDatabase(): Database.Database {
    return this.db;
  }
}
