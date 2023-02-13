use std::fs::create_dir;

macro_rules! bench_create_internal {
    ($name:literal, $tasks:literal) => {
        std::process::Command::new(concat!("C/build/benches/", $name))
        .arg(concat!("C/output/", $name, "/", $name, "_", stringify!($tasks), "_.csv"))
        .arg(stringify!($tasks))
    };
}

macro_rules! bench_internal {
    ($name:literal, $tasks:literal) => {
        bench_create_internal!($name, $tasks)
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
    };
}

macro_rules! bench_nt_internal {
    ($name:literal, $tasks:literal) => {
        bench_create_internal!($name, $tasks)
        .env("MPI_USE_NT", "1")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
    };
}

macro_rules! bench_create {
    ($name:literal, $out:literal, $job:literal, $tasks:literal) => {
        std::process::Command::new("sbatch")
            .arg("--exclusive")
            .arg(concat!("--ntasks-per-node=", stringify!($tasks)))
            .arg(concat!("C/jobs/", $job, ".job"))
            .arg(concat!("C/build/benches/", $name))
            .arg(concat!("C/output/", $name, "/", $out))
            .arg("0")
    };
}

macro_rules! bench {
    ($name:literal, $out:literal, $job:literal, $tasks:literal) => {
        let _ = create_dir(concat!("C/output/", $name));
        bench_create!($name, $out, $job, $tasks)
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
    };
}

macro_rules! bench_nt {
    ($name:literal, $out:expr, $job:literal, $tasks:literal) => {
        let _ = create_dir(concat!("C/output/", $name));
        bench_create!($name, $out, $job, $tasks)
            .env("MPI_USE_NT", "1")
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
    };
}

fn bench_slurm()
{
    bench!("bcast", "bcast_4_.csv", "collective", 4);
    bench_nt!("bcast", "ntbcast_4_.csv", "collective", 4);
    bench!("bcast", "bcast_8_.csv", "collective", 8);
    bench_nt!("bcast", "ntbcast_8_.csv", "collective", 8);

    bench!("gather", "gather_4_.csv", "collective", 4);
    bench_nt!("gather", "ntgather_4_.csv", "collective", 4);
    bench!("gather", "gather_8_.csv", "collective", 8);
    bench_nt!("gather", "ntgather_8_.csv", "collective", 8);

    bench!("allgather", "allgather_4_.csv", "collective", 4);
    bench_nt!("allgather", "ntallgather_4_.csv", "collective", 4);
    bench!("allgather", "allgather_8_.csv", "collective", 8);
    bench_nt!("allgather", "ntallgather_8_.csv", "collective", 8);

    bench!("allreduce", "allreduce_4_.csv", "collective", 4);
    bench_nt!("allreduce", "ntallreduce_4_.csv", "collective", 4);
    bench!("allreduce", "allreduce_8_.csv", "collective", 8);
    bench_nt!("allreduce", "ntallreduce_8_.csv", "collective", 8);

    bench!("reduce", "reduce_4_.csv", "collective", 4);
    bench_nt!("reduce", "ntreduce_4_.csv", "collective", 4);
    bench!("reduce", "reduce_8_.csv", "collective", 8);
    bench_nt!("reduce", "ntreduce_8_.csv", "collective", 8);
}

fn bench_internal()
{
    bench_internal!("bcast", 4);
    bench_internal!("bcast", 8);
    bench_nt_internal!("bcast", 4);
    bench_nt_internal!("bcast", 8);

    bench_internal!("gather", 4);
    bench_internal!("gather", 8);
    bench_nt_internal!("gather", 4);
    bench_nt_internal!("gather", 8);

    bench_internal!("reduce", 4);
    bench_internal!("reduce", 8);
    bench_nt_internal!("reduce", 4);
    bench_nt_internal!("reduce", 8);

    bench_internal!("allgather", 4);
    bench_internal!("allgather", 8);
    bench_nt_internal!("allgather", 4);
    bench_nt_internal!("allgather", 8);

    bench_internal!("allreduce", 4);
    bench_internal!("allreduce", 8);
    bench_nt_internal!("allreduce", 4);
    bench_nt_internal!("allreduce", 8);
}

fn main() {
    let _ = create_dir("C/output");

    if let Ok(_) = std::process::Command::new("sbatch").arg("-V").spawn() {
        bench_slurm();
    } else {
        bench_internal();
    }
}
