use cp_anywhere::batch::run_batch;
use std::num::NonZeroUsize;
use std::sync::Barrier;

#[test]
fn serial_batch_runs_in_the_caller() {
    let caller = std::thread::current().id();
    let result = run_batch(&[7, 2, 1], NonZeroUsize::MIN, |&input| {
        assert_eq!(std::thread::current().id(), caller);
        input * 10
    });
    assert_eq!(result, [70, 20, 10]);
}

#[test]
fn two_workers_process_each_job_once_and_return_input_order() {
    let start = Barrier::new(2);
    let caller = std::thread::current().id();
    let result = run_batch(&[7, 2, 1, 3], NonZeroUsize::new(2).unwrap(), |&input| {
        assert_ne!(std::thread::current().id(), caller);
        if input == 7 || input == 2 {
            start.wait();
        }
        input * 10
    });
    assert_eq!(result, [70, 20, 10, 30]);
}
