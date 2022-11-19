use cmake::Config;

fn main()
{
    Config::new(".").profile("Release").build_target("all").out_dir(".").build();
}