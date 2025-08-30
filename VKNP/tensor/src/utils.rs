/// Computes the strides for an empty tensor given its shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![0; n];
    if n == 0 {
        return strides;
    }
    // The last dimension has stride 1
    strides[n - 1] = 1;
    // We go back from the penultimate (n-2) to the 0th
    for i in (0..n-1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}


#[cfg(test)]
mod tests {
    use super::compute_strides;

    #[test]
    fn test_compute_strides_simple() {
        assert_eq!(compute_strides(&[]), vec![]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[4, 1, 5]), vec![5, 5, 1]);
    }
}