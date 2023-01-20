pub struct KeyChanger<'a> {
    group: &'a mut crate::CommGroup,
    comm: crate::MPI_Comm,
}

impl<'a> KeyChanger<'a> {
    pub fn new(group: &'a mut crate::CommGroup, comm: crate::MPI_Comm) -> Self {
        group.inc_key(comm);
        KeyChanger { group, comm }
    }
}

impl<'a> Drop for KeyChanger<'a> {
    fn drop(&mut self) {
        self.group.dec_key(self.comm);
    }
}