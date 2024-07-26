# Block-Scaffold-GS

## Motivation

In this project, we found that GS-based methods often result in blurred images when rendering large-scale scenes or aerial and street fusion datasets. One reason is that densification fails when handling numerous images or different classes of data using the original parameters. To address this, we employ a "divide-and-conquer" approach similar to hierarchical-3DGS and Vast-Gaussian. Leveraging the power of Scaffold-GS, we reconstruct each chunk individually. To prevent artifacts at the edges of each chunk, we optimize the entire set of chunks after optimizing each one individually.