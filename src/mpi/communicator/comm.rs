use crate::{shared::*, types::*};

#[derive(Clone)]
pub(super) struct Comm {
    pub prank: Vec<i32>,
    pub errh: MPI_Errhandler,
    pub rank: i32,
    pub key: i32,
}

impl Comm {
    pub const fn new() -> Comm {
        return Comm {
            prank: Vec::new(),
            errh: 0,
            rank: 0,
            key: 0,
        };
    }
}

impl Default for Comm {
    fn default() -> Self {
        return Self::new();
    }
}

unsafe impl Sync for Comm {}

#[derive(Clone, Copy)]
pub struct CommSplit {
    pub col: i32,
    pub key: i32,
    pub rank: i32,
    pub grank: i32,
    pub key_max: i32,
}

impl CommSplit {
    pub const fn new() -> Self {
        uninit()
    }
}
