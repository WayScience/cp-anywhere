//! Bounded field-level execution for native Rust callers.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Run an atomic operation once per input, with at most `workers` concurrent calls.
/// Results are collected in input order. For per-field errors, return a `Result`
/// from `operation`; all inputs are still processed. A worker panic propagates.
/// With one worker or fewer than two inputs, run on the caller's thread.
pub fn run_batch<T: Sync, R: Send>(
    inputs: &[T],
    workers: NonZeroUsize,
    operation: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let workers = workers.get().min(inputs.len());
    if workers <= 1 {
        return inputs.iter().map(operation).collect();
    }

    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        let operation = &operation;
        let next = &next;
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(move || {
                    let mut results = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(input) = inputs.get(index) else {
                            break;
                        };
                        results.push((index, operation(input)));
                    }
                    results
                })
            })
            .collect();
        let mut results: Vec<_> = handles
            .into_iter()
            .flat_map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|error| std::panic::resume_unwind(error))
            })
            .collect();
        results.sort_unstable_by_key(|(index, _)| *index);
        results.into_iter().map(|(_, output)| output).collect()
    })
}
