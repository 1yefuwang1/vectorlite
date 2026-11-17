#ifndef VECTORLITE_BATCH_H_
#define VECTORLITE_BATCH_H_

#include <stdint.h>

/* sqlite3_bind_pointer's type tag is this NUL-terminated string literal. */
#define VECTORLITE_BATCH_F32_V1_TAG "vectorlite.batch.f32.v1"
#define VECTORLITE_BATCH_F32_V1_ABI_VERSION 1u
/* sizeof(BatchF32V1) on every official 64-bit release target. */
#define VECTORLITE_BATCH_F32_V1_STRUCT_SIZE 40u

/*
 * Native batch INSERT input, ABI version 1.
 *
 * Bind a pointer to THIS DESCRIPTOR using sqlite3_bind_pointer, not a pointer to
 * its vectors, not an INTEGER address, and not a BLOB containing an address.
 * No Python API or linked SQLite library is required by this header. The native
 * application includes its own SQLite declarations to prepare/bind/step SQL.
 *
 * Native caller safety contract:
 * - The descriptor is a readable, naturally aligned, fully initialized object
 *   of this exact layout. abi_version is 1 and struct_size is sizeof(BatchF32V1).
 * - dimension is positive and matches the receiving vector column. count fits
 *   the nonnegative signed 64-bit public count domain. Zero rows are permitted.
 * - vectors addresses count * dimension consecutive native IEEE-754 float32
 *   elements in row-major order; rowids addresses count explicit signed 64-bit
 *   IDs. Each nonempty array lies entirely in one readable, correctly aligned
 *   allocation. The extension rejects negative rowids; zero is permitted.
 * - All size products fit both size_t and the positive ptrdiff_t address-span
 *   domain. No addressed range may wrap the native address space.
 * - The descriptor AND BOTH BUFFERS stay alive, readable and IMMUTABLE until
 *   sqlite3_clear_bindings, replacement of this binding, or statement finalize
 *   releases the binding. sqlite3_reset alone DOES NOT release bindings.
 * - Do not mutate/free these objects from another thread or a reentrant callback.
 *   The extension borrows them only during a SQLite callback, copies bounded
 *   owned chunks for insertion, and never retains the borrowed view in a task.
 *
 * For count == 0 the array pointers may be NULL, but the descriptor must still
 * be valid and dimension must still match. Runtime metadata, null, alignment
 * and arithmetic checks cannot prove allocation validity or buffer length.
 * A matching tag is a type label, NOT pointer authentication: binding an
 * arbitrary address with this tag violates the native caller contract and can
 * cause undefined behavior. SQL integers are never interpreted as pointers.
 *
 * Example (stmt is an already prepared native batch INSERT statement):
 *
 *   float vectors[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
 *   int64_t rowids[2] = {10, 20};
 *   BatchF32V1 batch = {
 *       VECTORLITE_BATCH_F32_V1_ABI_VERSION, (uint32_t)sizeof(BatchF32V1),
 *       2, 3, vectors, rowids};
 *   int rc = sqlite3_bind_pointer(stmt, 1, &batch,
 *                                VECTORLITE_BATCH_F32_V1_TAG, NULL);
 *   // Check rc, step the statement, then finalize it BEFORE these locals die.
 *   // A reset is insufficient: explicitly clear/rebind/finalize the binding.
 *
 * NULL destructor leaves ownership with the caller. A native destructor may
 * instead own the descriptor and both allocations; it must release them only
 * when SQLite releases the binding, including on binding failure as required
 * by sqlite3_bind_pointer. Never attach a freeing destructor to stack storage.
 */
typedef struct BatchF32V1 {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t count;
  uint64_t dimension;
  const float* vectors;
  const int64_t* rowids;
} BatchF32V1;

#endif /* VECTORLITE_BATCH_H_ */
