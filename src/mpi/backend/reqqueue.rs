use crate::shared::*;
use std::ptr::NonNull;

pub struct Queue<T, const N: usize> {
    queue: [T; N],
    flags: [bool; N],
    size: usize,
    head: usize,
    tail: usize,
}

impl<T, const N: usize> Queue<T, N>
where
    T: Clone + Copy + Default,
{
    pub fn new() -> Self {
        Queue::<T, N> {
            queue: [Default::default(); N],
            flags: [false; N],
            size: 0,
            head: 0,
            tail: 0,
        }
    }

    pub const fn new_val(val: T) -> Self {
        Queue::<T, N> {
            queue: [val; N],
            flags: [false; N],
            size: 0,
            head: 0,
            tail: 0,
        }
    }

    pub fn push(&mut self) -> Option<&mut T> {
        if self.size == 0 {
            self.head = 0;
            self.tail = 0;
            self.size += 1;
            self.flags[0] = true;
            return Some(&mut self.queue[0]);
        } else if self.size < N {
            self.tail = (self.tail + 1) % N;
            self.flags[self.tail] = true;
            self.size += 1;
            return Some(&mut self.queue[self.tail]);
        }

        None
    }

    pub fn erase(&mut self, mut i: usize) {
        if self.size == 0 {
            return;
        }

        if self.flags[i] {
            self.flags[i] = false;

            if i == self.head {
                loop {
                    i = (i + 1) % N;
                    if self.flags[i] {
                        self.head = i;
                        break;
                    }

                    if i == self.tail {
                        break;
                    }
                }
            } else if i == self.tail {
                loop {
                    i = (N + i - 1) % N;
                    if self.flags[i] {
                        self.tail = i;
                        break;
                    }
                    if i == self.head {
                        break;
                    }
                }
            }
        }
        self.size -= 1;
    }

    #[inline(always)]
    pub fn erase_ptr(&mut self, ptr: *const T) {
        unsafe { self.erase(ptr.offset_from(self.queue.as_ptr()) as usize) }
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<'_, T, N> {
        IterMut::new(self)
    }

    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, T, N> {
        Iter::new(self)
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

impl<T, const N: usize> Default for Queue<T, N>
where
    T: Copy + Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, T, const N: usize> {
    q: &'a Queue<T, N>,
    item: Option<&'a T>,
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.item.is_none() {
            return None;
        }

        let res = self.item;
        if self.q.size > 1 {
            let mut idx = unsafe {
                (res.unwrap_unchecked() as *const T).offset_from(self.q.queue.as_ptr()) as usize
            };
            loop {
                idx = (idx + 1) % N;
                if self.q.tail == idx {
                    self.item = None;
                    break;
                }

                if self.q.flags[idx] {
                    self.item = Some(&self.q.queue[idx]);
                    break;
                }
            }
        } else {
            self.item = None;
        }

        res
    }
}

pub struct IterMut<'a, T, const N: usize> {
    pq: NonNull<Queue<T, N>>,
    item: Option<&'a mut T>,
}

impl<'a, T, const N: usize> Iterator for IterMut<'a, T, N>
where
    T: Copy,
{
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.item.is_none() {
            return None;
        }

        let res = unsafe { *self.item.as_mut().unwrap_unchecked() as *mut T };
        let q = unsafe { self.pq.as_mut() };
        if q.size > 1 {
            let mut idx = unsafe { res.offset_from(q.queue.as_ptr()) as usize };

            loop {
                idx = (idx + 1) % N;
                if q.head == idx {
                    self.item = None;
                    break;
                }

                if q.flags[idx] {
                    self.item = Some(&mut q.queue[idx]);
                    break;
                }
            }
        } else {
            self.item = None;
        }

        return unsafe { Some(&mut *res) };
    }
}

impl<'a, T, const N: usize> IterMut<'a, T, N> {
    fn new(pq: *mut Queue<T, N>) -> Self {
        let first;
        let q = unsafe { &mut *pq };

        if q.size > 0 {
            first = Some(&mut q.queue[q.head]);
        } else {
            first = None;
        }

        IterMut {
            pq: unsafe { NonNull::new_unchecked(pq) },
            item: first,
        }
    }
}

impl<'a, T, const N: usize> Iter<'a, T, N> {
    fn new(q: &'a Queue<T, N>) -> Self {
        let first;
        if q.size > 0 {
            first = Some(&q.queue[q.head]);
        } else {
            first = None;
        }
        Iter { q, item: first }
    }
}

pub type RequestQueue = Queue<P_MPI_Request, 16>;

impl RequestQueue {
    pub fn find_by_tag(&mut self, rank: i32, tag: i32) -> MPI_Request {
        let val = self.iter_mut().find(|x| x.rank == rank && x.tag == tag);
        return if val.is_none() {
            null_mut()
        } else {
            unsafe { val.unwrap_unchecked() }
        };
    }

    #[inline(always)]
    pub fn contains(&self, req: MPI_Request) -> bool {
        return self
            .iter()
            .find(|&x| x as *const P_MPI_Request == req)
            .is_some();
    }

    pub const fn new_c() -> Self {
        RequestQueue::new_val(P_MPI_Request::new())
    }
}
