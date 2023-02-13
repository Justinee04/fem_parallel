import fem_laplacian as laplacian

a = laplacian.compute_elementary_matrix( (0.,0.), (1.,0.), (0.,1.) )
print(a)

