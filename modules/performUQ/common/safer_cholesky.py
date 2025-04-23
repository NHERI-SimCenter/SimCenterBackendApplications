"""
safer_cholesky.py.

Provides a robust and configurable Cholesky decomposition utility that handles
non-positive-definite matrices using jitter regularization and eigenvalue
repair fallback. Optionally logs diagnostics and dumps matrices to disk
for later inspection.

Typical usage:

    from safer_cholesky import SaferCholesky
    chol = SaferCholesky(debug=True)
    L = chol.decompose(matrix, matrix_id="my_matrix")
"""

# import logging
from pathlib import Path

import numpy as np


class SaferCholesky:
    """
    Robust Cholesky decomposition with diagnostic logging and fallback strategies.

    Parameters
    ----------
        debug (bool): Enable debug logging and matrix dumping. Default is False.
        dump_dir (str): Directory to save matrices for failed decompositions.
    """

    def __init__(self, *, debug=False, dump_dir='../matrix_dumps'):
        self.debug = debug
        self.dump_dir = dump_dir

        if self.debug:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

        # logging.basicConfig(
        #     level=logging.DEBUG if self.debug else logging.INFO,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     filename='tmcmc_debug.log',
        #     filemode='w',
        # )

    def decompose(
        self,
        matrix,
        matrix_id='cov',
        max_attempts=5,
        jitter_start=1e-10,
        eig_clip_threshold=1e-8,
    ):
        """
        Attempt Cholesky decomposition, adding jitter and using EVD repair as fallback.

        Parameters
        ----------
            matrix (np.ndarray): The matrix to decompose.
            matrix_id (str): Identifier used in logs and saved filenames.
            max_attempts (int): Max jitter retries before EVD fallback.
            jitter_start (float): Starting jitter value to add to the diagonal.
            eig_clip_threshold (float): Min eigenvalue allowed during EVD repair.

        Returns
        -------
            np.ndarray: Lower triangular matrix from Cholesky decomposition.

        Raises
        ------
            RuntimeError: If decomposition fails even after all fallbacks.
        """
        jitter = jitter_start
        matrix = (matrix + matrix.T) / 2  # Ensure symmetry

        for attempt in range(max_attempts):
            try:
                return np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:  # noqa: PERF203
                # min_eig = np.min(np.linalg.eigvalsh(matrix))
                # cond_num = np.linalg.cond(matrix)
                # logging.warning(
                #     f'[{matrix_id}] Attempt {attempt+1}: Cholesky failed. '
                #     f'min_eigenvalue={min_eig:.2e}, cond_num={cond_num:.2e}, jitter={jitter:.1e}'
                # )
                matrix += jitter * np.eye(matrix.shape[0])
                jitter *= 10

        # logging.warning(f'[{matrix_id}] Falling back to EVD-based repair.')
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals_clipped = np.clip(eigvals, eig_clip_threshold, None)
        repaired_matrix = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        try:
            return np.linalg.cholesky(repaired_matrix)
        except np.linalg.LinAlgError as exc:
            # logging.exception(
            #     f'[{matrix_id}] Final Cholesky failed even after EVD repair.'
            # )

            if self.debug:
                np.save(f'{self.dump_dir}/{matrix_id}_original.npy', matrix)
                np.save(f'{self.dump_dir}/{matrix_id}_eigvals.npy', eigvals)
                np.save(
                    f'{self.dump_dir}/{matrix_id}_eigvals_clipped.npy',
                    eigvals_clipped,
                )
                np.save(f'{self.dump_dir}/{matrix_id}_repaired.npy', repaired_matrix)
                # logging.debug(
                #     f'[{matrix_id}] Matrix and eigenvalues saved to {self.dump_dir}/'
                # )

            msg = f'Cholesky failed for matrix {matrix_id}'
            raise RuntimeError(msg) from exc
