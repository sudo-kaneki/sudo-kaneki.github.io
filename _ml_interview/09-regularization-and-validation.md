---
title: "Regularization and Validation"
chapter: 9
permalink: "/blog/cracking-the-ml-interview/regularization-and-validation/"
updated: 2026-11-01
description: "Regularization makes a model simpler than the data alone would make it, to trade a little bias for a large drop in variance."
---

Regularization makes a model simpler than the data alone would make it, to trade a little bias for a large drop in variance. Validation tells you whether the trade worked. The two go together: every regularization method has a strength to tune, and cross-validation is how you tune it honestly. Expect questions on L1 versus L2, dropout, and how to validate without fooling yourself.

## 9.1 Intuition

**Why regularize?** A flexible model can fit the training data, noise included. Regularization adds a preference for simpler solutions (small weights, few features, redundant units), which lowers variance at the cost of a little bias. The strength is a hyperparameter, usually written λ or α: zero means no regularization, and very large values give a model too simple to be useful.

**L2 regularization (ridge regression, weight decay)** (reference 19) adds λ times the sum of squared weights to the loss. It shrinks all weights smoothly toward zero but rarely makes any exactly zero. With correlated features, it spreads the weight evenly across them instead of picking one. In neural networks the same penalty is called **weight decay**, because each update multiplies the weights by a factor slightly below 1.

**L1 regularization (the lasso, Least Absolute Shrinkage and Selection Operator)** adds λ times the sum of absolute weights. It drives many weights to **exactly zero**, so it performs **feature selection** and gives a sparse, more interpretable model. With a group of correlated features it tends to keep one and drop the rest, somewhat arbitrarily.

**Elastic net** combines both penalties. It keeps L1's sparsity while L2 makes it share weight within groups of correlated features and stabilizes the selection.

**Why L1 gives zeros and L2 does not: the geometry.** Regularizing is equivalent to minimizing the training error subject to a budget on the weights. The L1 budget is a diamond with corners on the axes; the L2 budget is a circle. The solution is where the error contours first touch the budget region, which for the diamond is very often a corner, where some coefficients are exactly zero.

<figure class="diagram">
<div class="diagram__scroll">
<svg viewBox="0 0 760 400" style="--dg-w: 760" role="img" aria-labelledby="ch9-geo-title ch9-geo-desc"><title id="ch9-geo-title">Constraint geometry of the lasso and ridge regression</title><desc id="ch9-geo-desc">Two panels with axes beta 1 and beta 2. In each, elliptical contours of the residual sum of squares surround the least-squares estimate in the upper right. Left: the lasso budget is a diamond with corners on the axes, and the first contour to touch it meets the top corner, where beta 1 is exactly zero. Right: the ridge budget is a circle, and the first contour touches it at a point where both coefficients are non-zero.</desc><defs><clipPath id="ch9-geo-clip-0"><rect x="60" y="40" width="275" height="302"/></clipPath><clipPath id="ch9-geo-clip-1"><rect x="440" y="40" width="275" height="302"/></clipPath></defs><text class="dg-strong" x="197.5" y="26" text-anchor="middle">lasso: |β₁| + |β₂| ≤ s</text><rect class="dg-box" x="60" y="40" width="275" height="302"/><g clip-path="url(#ch9-geo-clip-0)"><polygon class="dg-shade" points="197.5,260.0 142.5,205.0 87.5,260.0 142.5,315.0"/><polygon class="dg-line" points="197.5,260.0 142.5,205.0 87.5,260.0 142.5,315.0"/><line class="dg-rule" x1="60.0" y1="260.0" x2="335.0" y2="260.0"/><line class="dg-rule" x1="142.5" y1="342.5" x2="142.5" y2="40.0"/><polyline class="dg-line" points="231.4,106.8 230.5,106.1 229.5,105.3 228.6,104.6 227.6,103.9 226.7,103.2 225.7,102.6 224.7,101.9 223.7,101.3 222.7,100.7 221.7,100.1 220.7,99.5 219.7,99.0 218.7,98.4 217.7,97.9 216.7,97.4 215.6,97.0 214.6,96.5 213.6,96.1 212.6,95.7 211.5,95.3 210.5,94.9 209.5,94.6 208.5,94.3 207.5,94.0 206.5,93.7 205.5,93.5 204.5,93.2 203.5,93.0 202.5,92.9 201.5,92.7 200.6,92.6 199.6,92.5 198.7,92.4 197.7,92.3 196.8,92.3 195.9,92.3 195.0,92.3 194.1,92.4 193.2,92.4 192.4,92.5 191.5,92.7 190.7,92.8 189.9,93.0 189.1,93.1 188.3,93.4 187.5,93.6 186.8,93.9 186.1,94.1 185.4,94.5 184.7,94.8 184.0,95.1 183.4,95.5 182.7,95.9 182.1,96.3 181.5,96.8 181.0,97.2 180.5,97.7 179.9,98.2 179.4,98.7 179.0,99.3 178.5,99.9 178.1,100.4 177.7,101.0 177.4,101.7 177.0,102.3 176.7,103.0 176.4,103.6 176.1,104.3 175.9,105.0 175.7,105.8 175.5,106.5 175.3,107.2 175.2,108.0 175.1,108.8 175.0,109.6 174.9,110.4 174.9,111.2 174.9,112.0 174.9,112.9 175.0,113.7 175.1,114.6 175.2,115.5 175.3,116.3 175.5,117.2 175.6,118.1 175.8,119.0 176.1,119.9 176.3,120.9 176.6,121.8 176.9,122.7 177.3,123.6 177.7,124.5 178.0,125.5 178.5,126.4 178.9,127.3 179.4,128.3 179.8,129.2 180.4,130.1 180.9,131.1 181.4,132.0 182.0,132.9 182.6,133.9 183.2,134.8 183.9,135.7 184.5,136.6 185.2,137.5 185.9,138.4 186.6,139.3 187.4,140.2 188.1,141.0 188.9,141.9 189.7,142.8 190.5,143.6 191.4,144.4 192.2,145.3 193.1,146.1 193.9,146.9 194.8,147.7 195.7,148.4 196.6,149.2 197.5,149.9 198.5,150.7 199.4,151.4 200.4,152.1 201.3,152.8 202.3,153.4 203.3,154.1 204.3,154.7 205.3,155.3 206.3,155.9 207.3,156.5 208.3,157.0 209.3,157.6 210.3,158.1 211.3,158.6 212.4,159.0 213.4,159.5 214.4,159.9 215.4,160.3 216.5,160.7 217.5,161.1 218.5,161.4 219.5,161.7 220.5,162.0 221.5,162.3 222.5,162.5 223.5,162.8 224.5,163.0 225.5,163.1 226.5,163.3 227.4,163.4 228.4,163.5 229.3,163.6 230.3,163.7 231.2,163.7 232.1,163.7 233.0,163.7 233.9,163.6 234.8,163.6 235.6,163.5 236.5,163.3 237.3,163.2 238.1,163.0 238.9,162.9 239.7,162.6 240.5,162.4 241.2,162.1 241.9,161.9 242.6,161.5 243.3,161.2 244.0,160.9 244.6,160.5 245.3,160.1 245.9,159.7 246.5,159.2 247.0,158.8 247.5,158.3 248.1,157.8 248.6,157.3 249.0,156.7 249.5,156.1 249.9,155.6 250.3,155.0 250.6,154.3 251.0,153.7 251.3,153.0 251.6,152.4 251.9,151.7 252.1,151.0 252.3,150.2 252.5,149.5 252.7,148.8 252.8,148.0 252.9,147.2 253.0,146.4 253.1,145.6 253.1,144.8 253.1,144.0 253.1,143.1 253.0,142.3 252.9,141.4 252.8,140.5 252.7,139.7 252.5,138.8 252.4,137.9 252.2,137.0 251.9,136.1 251.7,135.1 251.4,134.2 251.1,133.3 250.7,132.4 250.3,131.5 250.0,130.5 249.5,129.6 249.1,128.7 248.6,127.7 248.2,126.8 247.6,125.9 247.1,124.9 246.6,124.0 246.0,123.1 245.4,122.1 244.8,121.2 244.1,120.3 243.5,119.4 242.8,118.5 242.1,117.6 241.4,116.7 240.6,115.8 239.9,115.0 239.1,114.1 238.3,113.2 237.5,112.4 236.6,111.6 235.8,110.7 234.9,109.9 234.1,109.1 233.2,108.3 232.3,107.6 231.4,106.8" stroke-dasharray="4 3"/><polyline class="dg-line" points="246.5,88.3 244.8,86.9 243.0,85.6 241.3,84.2 239.5,82.9 237.7,81.7 235.9,80.4 234.0,79.2 232.2,78.0 230.3,76.9 228.5,75.8 226.6,74.7 224.7,73.7 222.8,72.7 220.9,71.7 219.0,70.8 217.1,69.9 215.1,69.1 213.2,68.3 211.3,67.5 209.4,66.8 207.5,66.1 205.6,65.5 203.7,64.9 201.8,64.3 199.9,63.8 198.0,63.4 196.2,63.0 194.3,62.6 192.5,62.3 190.7,62.0 188.9,61.7 187.1,61.5 185.3,61.4 183.5,61.3 181.8,61.2 180.1,61.2 178.4,61.3 176.8,61.4 175.1,61.5 173.5,61.7 171.9,61.9 170.4,62.1 168.9,62.4 167.4,62.8 165.9,63.2 164.5,63.6 163.1,64.1 161.7,64.7 160.4,65.2 159.1,65.9 157.9,66.5 156.7,67.2 155.5,68.0 154.4,68.7 153.3,69.6 152.2,70.4 151.2,71.3 150.3,72.3 149.4,73.3 148.5,74.3 147.7,75.3 146.9,76.4 146.1,77.6 145.4,78.7 144.8,79.9 144.2,81.2 143.7,82.4 143.2,83.7 142.7,85.0 142.3,86.4 141.9,87.8 141.6,89.2 141.4,90.6 141.2,92.1 141.0,93.6 140.9,95.1 140.9,96.6 140.9,98.2 140.9,99.7 141.0,101.3 141.1,102.9 141.3,104.6 141.6,106.2 141.9,107.9 142.2,109.5 142.6,111.2 143.1,112.9 143.6,114.6 144.1,116.3 144.7,118.1 145.3,119.8 146.0,121.5 146.7,123.3 147.5,125.0 148.3,126.8 149.2,128.5 150.1,130.3 151.0,132.0 152.0,133.8 153.1,135.5 154.2,137.2 155.3,139.0 156.4,140.7 157.6,142.4 158.9,144.1 160.2,145.8 161.5,147.5 162.8,149.1 164.2,150.8 165.6,152.4 167.1,154.0 168.6,155.6 170.1,157.2 171.6,158.8 173.2,160.3 174.8,161.8 176.4,163.3 178.1,164.8 179.8,166.2 181.5,167.7 183.2,169.1 185.0,170.4 186.7,171.8 188.5,173.1 190.3,174.3 192.1,175.6 194.0,176.8 195.8,178.0 197.7,179.1 199.5,180.2 201.4,181.3 203.3,182.3 205.2,183.3 207.1,184.3 209.0,185.2 210.9,186.1 212.9,186.9 214.8,187.7 216.7,188.5 218.6,189.2 220.5,189.9 222.4,190.5 224.3,191.1 226.2,191.7 228.1,192.2 230.0,192.6 231.8,193.0 233.7,193.4 235.5,193.7 237.3,194.0 239.1,194.3 240.9,194.5 242.7,194.6 244.5,194.7 246.2,194.8 247.9,194.8 249.6,194.7 251.2,194.6 252.9,194.5 254.5,194.3 256.1,194.1 257.6,193.9 259.1,193.6 260.6,193.2 262.1,192.8 263.5,192.4 264.9,191.9 266.3,191.3 267.6,190.8 268.9,190.1 270.1,189.5 271.3,188.8 272.5,188.0 273.6,187.3 274.7,186.4 275.8,185.6 276.8,184.7 277.7,183.7 278.6,182.7 279.5,181.7 280.3,180.7 281.1,179.6 281.9,178.4 282.6,177.3 283.2,176.1 283.8,174.8 284.3,173.6 284.8,172.3 285.3,171.0 285.7,169.6 286.1,168.2 286.4,166.8 286.6,165.4 286.8,163.9 287.0,162.4 287.1,160.9 287.1,159.4 287.1,157.8 287.1,156.3 287.0,154.7 286.9,153.1 286.7,151.4 286.4,149.8 286.1,148.1 285.8,146.5 285.4,144.8 284.9,143.1 284.4,141.4 283.9,139.7 283.3,137.9 282.7,136.2 282.0,134.5 281.3,132.7 280.5,131.0 279.7,129.2 278.8,127.5 277.9,125.7 277.0,124.0 276.0,122.2 274.9,120.5 273.8,118.8 272.7,117.0 271.6,115.3 270.4,113.6 269.1,111.9 267.8,110.2 266.5,108.5 265.2,106.9 263.8,105.2 262.4,103.6 260.9,102.0 259.4,100.4 257.9,98.8 256.4,97.2 254.8,95.7 253.2,94.2 251.6,92.7 249.9,91.2 248.2,89.8 246.5,88.3" stroke-dasharray="4 3"/><polyline class="dg-line" points="261.6,69.9 259.1,67.9 256.5,65.9 253.9,63.9 251.3,62.0 248.7,60.2 246.0,58.4 243.3,56.6 240.6,54.9 237.9,53.2 235.2,51.6 232.4,50.0 229.6,48.5 226.9,47.0 224.1,45.6 221.3,44.3 218.5,43.0 215.7,41.7 212.9,40.6 210.1,39.5 207.3,38.4 204.5,37.4 201.7,36.5 198.9,35.6 196.1,34.8 193.4,34.1 190.6,33.4 187.9,32.8 185.2,32.2 182.5,31.8 179.8,31.3 177.2,31.0 174.6,30.7 172.0,30.5 169.4,30.4 166.9,30.3 164.4,30.3 161.9,30.3 159.5,30.4 157.1,30.6 154.7,30.9 152.4,31.2 150.2,31.6 147.9,32.0 145.7,32.6 143.6,33.1 141.5,33.8 139.5,34.5 137.5,35.3 135.6,36.1 133.7,37.0 131.9,38.0 130.1,39.0 128.4,40.1 126.7,41.3 125.1,42.5 123.6,43.7 122.1,45.1 120.7,46.4 119.4,47.9 118.1,49.4 116.9,50.9 115.7,52.5 114.7,54.2 113.6,55.9 112.7,57.6 111.8,59.4 111.0,61.3 110.3,63.2 109.6,65.1 109.0,67.1 108.5,69.1 108.1,71.2 107.7,73.3 107.4,75.4 107.2,77.6 107.0,79.8 106.9,82.0 106.9,84.3 107.0,86.6 107.1,88.9 107.4,91.3 107.6,93.7 108.0,96.1 108.4,98.5 108.9,101.0 109.5,103.4 110.2,105.9 110.9,108.4 111.7,110.9 112.5,113.5 113.5,116.0 114.5,118.5 115.5,121.1 116.7,123.6 117.9,126.2 119.1,128.8 120.5,131.3 121.8,133.9 123.3,136.4 124.8,139.0 126.4,141.5 128.0,144.0 129.7,146.6 131.5,149.1 133.3,151.6 135.2,154.0 137.1,156.5 139.1,158.9 141.1,161.3 143.2,163.7 145.3,166.1 147.5,168.4 149.7,170.8 152.0,173.0 154.3,175.3 156.6,177.5 159.0,179.7 161.4,181.9 163.9,184.0 166.4,186.1 168.9,188.1 171.5,190.1 174.1,192.1 176.7,194.0 179.3,195.8 182.0,197.6 184.7,199.4 187.4,201.1 190.1,202.8 192.8,204.4 195.6,206.0 198.4,207.5 201.1,209.0 203.9,210.4 206.7,211.7 209.5,213.0 212.3,214.3 215.1,215.4 217.9,216.5 220.7,217.6 223.5,218.6 226.3,219.5 229.1,220.4 231.9,221.2 234.6,221.9 237.4,222.6 240.1,223.2 242.8,223.8 245.5,224.2 248.2,224.7 250.8,225.0 253.4,225.3 256.0,225.5 258.6,225.6 261.1,225.7 263.6,225.7 266.1,225.7 268.5,225.6 270.9,225.4 273.3,225.1 275.6,224.8 277.8,224.4 280.1,224.0 282.3,223.4 284.4,222.9 286.5,222.2 288.5,221.5 290.5,220.7 292.4,219.9 294.3,219.0 296.1,218.0 297.9,217.0 299.6,215.9 301.3,214.7 302.9,213.5 304.4,212.3 305.9,210.9 307.3,209.6 308.6,208.1 309.9,206.6 311.1,205.1 312.3,203.5 313.3,201.8 314.4,200.1 315.3,198.4 316.2,196.6 317.0,194.7 317.7,192.8 318.4,190.9 319.0,188.9 319.5,186.9 319.9,184.8 320.3,182.7 320.6,180.6 320.8,178.4 321.0,176.2 321.1,174.0 321.1,171.7 321.0,169.4 320.9,167.1 320.6,164.7 320.4,162.3 320.0,159.9 319.6,157.5 319.1,155.0 318.5,152.6 317.8,150.1 317.1,147.6 316.3,145.1 315.5,142.5 314.5,140.0 313.5,137.5 312.5,134.9 311.3,132.4 310.1,129.8 308.9,127.2 307.5,124.7 306.2,122.1 304.7,119.6 303.2,117.0 301.6,114.5 300.0,112.0 298.3,109.4 296.5,106.9 294.7,104.4 292.8,102.0 290.9,99.5 288.9,97.1 286.9,94.7 284.8,92.3 282.7,89.9 280.5,87.6 278.3,85.2 276.0,83.0 273.7,80.7 271.4,78.5 269.0,76.3 266.6,74.1 264.1,72.0 261.6,69.9" stroke-dasharray="4 3"/><polyline class="dg-line-accent" points="280.5,46.8 277.0,44.0 273.4,41.2 269.8,38.5 266.2,35.8 262.5,33.2 258.7,30.6 255.0,28.2 251.2,25.8 247.4,23.4 243.6,21.2 239.7,19.0 235.9,16.8 232.0,14.8 228.1,12.8 224.2,11.0 220.3,9.2 216.3,7.4 212.4,5.8 208.5,4.2 204.6,2.8 200.7,1.4 196.8,0.1 192.9,-1.1 189.0,-2.3 185.2,-3.3 181.3,-4.2 177.5,-5.1 173.7,-5.9 170.0,-6.5 166.3,-7.1 162.6,-7.6 158.9,-8.0 155.3,-8.3 151.7,-8.5 148.1,-8.6 144.7,-8.6 141.2,-8.5 137.8,-8.4 134.5,-8.1 131.2,-7.8 127.9,-7.3 124.8,-6.8 121.6,-6.1 118.6,-5.4 115.6,-4.6 112.7,-3.7 109.8,-2.7 107.1,-1.6 104.4,-0.4 101.7,0.8 99.2,2.2 96.7,3.6 94.3,5.1 92.0,6.8 89.8,8.4 87.6,10.2 85.6,12.1 83.6,14.0 81.7,16.0 79.9,18.1 78.2,20.3 76.6,22.5 75.1,24.8 73.7,27.2 72.4,29.6 71.2,32.1 70.1,34.7 69.0,37.4 68.1,40.1 67.3,42.8 66.6,45.7 65.9,48.6 65.4,51.5 65.0,54.5 64.7,57.5 64.5,60.6 64.4,63.7 64.3,66.9 64.4,70.1 64.6,73.4 64.9,76.7 65.3,80.0 65.8,83.4 66.4,86.8 67.1,90.2 67.9,93.7 68.8,97.1 69.9,100.6 71.0,104.1 72.2,107.7 73.5,111.2 74.9,114.8 76.3,118.3 77.9,121.9 79.6,125.5 81.4,129.1 83.2,132.6 85.2,136.2 87.2,139.8 89.3,143.3 91.6,146.9 93.9,150.4 96.2,153.9 98.7,157.4 101.2,160.9 103.8,164.4 106.5,167.8 109.3,171.2 112.1,174.6 115.0,178.0 118.0,181.3 121.0,184.5 124.1,187.8 127.3,191.0 130.5,194.1 133.8,197.2 137.1,200.3 140.5,203.3 144.0,206.3 147.5,209.2 151.0,212.0 154.6,214.8 158.2,217.5 161.8,220.2 165.5,222.8 169.3,225.4 173.0,227.8 176.8,230.2 180.6,232.6 184.4,234.8 188.3,237.0 192.1,239.2 196.0,241.2 199.9,243.2 203.8,245.0 207.7,246.8 211.7,248.6 215.6,250.2 219.5,251.8 223.4,253.2 227.3,254.6 231.2,255.9 235.1,257.1 239.0,258.3 242.8,259.3 246.7,260.2 250.5,261.1 254.3,261.9 258.0,262.5 261.7,263.1 265.4,263.6 269.1,264.0 272.7,264.3 276.3,264.5 279.9,264.6 283.3,264.6 286.8,264.5 290.2,264.4 293.5,264.1 296.8,263.8 300.1,263.3 303.2,262.8 306.4,262.1 309.4,261.4 312.4,260.6 315.3,259.7 318.2,258.7 320.9,257.6 323.6,256.4 326.3,255.2 328.8,253.8 331.3,252.4 333.7,250.9 336.0,249.2 338.2,247.6 340.4,245.8 342.4,243.9 344.4,242.0 346.3,240.0 348.1,237.9 349.8,235.7 351.4,233.5 352.9,231.2 354.3,228.8 355.6,226.4 356.8,223.9 357.9,221.3 359.0,218.6 359.9,215.9 360.7,213.2 361.4,210.3 362.1,207.4 362.6,204.5 363.0,201.5 363.3,198.5 363.5,195.4 363.6,192.3 363.7,189.1 363.6,185.9 363.4,182.6 363.1,179.3 362.7,176.0 362.2,172.6 361.6,169.2 360.9,165.8 360.1,162.3 359.2,158.9 358.1,155.4 357.0,151.9 355.8,148.3 354.5,144.8 353.1,141.2 351.7,137.7 350.1,134.1 348.4,130.5 346.6,126.9 344.8,123.4 342.8,119.8 340.8,116.2 338.7,112.7 336.4,109.1 334.1,105.6 331.8,102.1 329.3,98.6 326.8,95.1 324.2,91.6 321.5,88.2 318.7,84.8 315.9,81.4 313.0,78.0 310.0,74.7 307.0,71.5 303.9,68.2 300.7,65.0 297.5,61.9 294.2,58.8 290.9,55.7 287.5,52.7 284.0,49.7 280.5,46.8"/></g><text class="dg-sub" x="329.0" y="254.0" text-anchor="end">β₁</text><text class="dg-sub" x="148.5" y="56.0">β₂</text><circle class="dg-dot" cx="214.0" cy="128.0" r="4"/><text class="dg-strong" x="222.0" y="132.0">β̂</text><circle class="dg-pt-a" cx="142.5" cy="205.0" r="5"/><text class="dg-key" x="68" y="193.0">β₁ = 0 here</text><text class="dg-strong" x="577.5" y="26" text-anchor="middle">ridge: β₁² + β₂² ≤ s²</text><rect class="dg-box" x="440" y="40" width="275" height="302"/><g clip-path="url(#ch9-geo-clip-1)"><circle class="dg-shade" cx="522.5" cy="260.0" r="55.0"/><circle class="dg-line" cx="522.5" cy="260.0" r="55.0"/><line class="dg-rule" x1="440.0" y1="260.0" x2="715.0" y2="260.0"/><line class="dg-rule" x1="522.5" y1="342.5" x2="522.5" y2="40.0"/><polyline class="dg-line" points="611.4,106.8 610.5,106.1 609.5,105.3 608.6,104.6 607.6,103.9 606.7,103.2 605.7,102.6 604.7,101.9 603.7,101.3 602.7,100.7 601.7,100.1 600.7,99.5 599.7,99.0 598.7,98.4 597.7,97.9 596.7,97.4 595.6,97.0 594.6,96.5 593.6,96.1 592.6,95.7 591.5,95.3 590.5,94.9 589.5,94.6 588.5,94.3 587.5,94.0 586.5,93.7 585.5,93.5 584.5,93.2 583.5,93.0 582.5,92.9 581.5,92.7 580.6,92.6 579.6,92.5 578.7,92.4 577.7,92.3 576.8,92.3 575.9,92.3 575.0,92.3 574.1,92.4 573.2,92.4 572.4,92.5 571.5,92.7 570.7,92.8 569.9,93.0 569.1,93.1 568.3,93.4 567.5,93.6 566.8,93.9 566.1,94.1 565.4,94.5 564.7,94.8 564.0,95.1 563.4,95.5 562.7,95.9 562.1,96.3 561.5,96.8 561.0,97.2 560.5,97.7 559.9,98.2 559.4,98.7 559.0,99.3 558.5,99.9 558.1,100.4 557.7,101.0 557.4,101.7 557.0,102.3 556.7,103.0 556.4,103.6 556.1,104.3 555.9,105.0 555.7,105.8 555.5,106.5 555.3,107.2 555.2,108.0 555.1,108.8 555.0,109.6 554.9,110.4 554.9,111.2 554.9,112.0 554.9,112.9 555.0,113.7 555.1,114.6 555.2,115.5 555.3,116.3 555.5,117.2 555.6,118.1 555.8,119.0 556.1,119.9 556.3,120.9 556.6,121.8 556.9,122.7 557.3,123.6 557.7,124.5 558.0,125.5 558.5,126.4 558.9,127.3 559.4,128.3 559.8,129.2 560.4,130.1 560.9,131.1 561.4,132.0 562.0,132.9 562.6,133.9 563.2,134.8 563.9,135.7 564.5,136.6 565.2,137.5 565.9,138.4 566.6,139.3 567.4,140.2 568.1,141.0 568.9,141.9 569.7,142.8 570.5,143.6 571.4,144.4 572.2,145.3 573.1,146.1 573.9,146.9 574.8,147.7 575.7,148.4 576.6,149.2 577.5,149.9 578.5,150.7 579.4,151.4 580.4,152.1 581.3,152.8 582.3,153.4 583.3,154.1 584.3,154.7 585.3,155.3 586.3,155.9 587.3,156.5 588.3,157.0 589.3,157.6 590.3,158.1 591.3,158.6 592.4,159.0 593.4,159.5 594.4,159.9 595.4,160.3 596.5,160.7 597.5,161.1 598.5,161.4 599.5,161.7 600.5,162.0 601.5,162.3 602.5,162.5 603.5,162.8 604.5,163.0 605.5,163.1 606.5,163.3 607.4,163.4 608.4,163.5 609.3,163.6 610.3,163.7 611.2,163.7 612.1,163.7 613.0,163.7 613.9,163.6 614.8,163.6 615.6,163.5 616.5,163.3 617.3,163.2 618.1,163.0 618.9,162.9 619.7,162.6 620.5,162.4 621.2,162.1 621.9,161.9 622.6,161.5 623.3,161.2 624.0,160.9 624.6,160.5 625.3,160.1 625.9,159.7 626.5,159.2 627.0,158.8 627.5,158.3 628.1,157.8 628.6,157.3 629.0,156.7 629.5,156.1 629.9,155.6 630.3,155.0 630.6,154.3 631.0,153.7 631.3,153.0 631.6,152.4 631.9,151.7 632.1,151.0 632.3,150.2 632.5,149.5 632.7,148.8 632.8,148.0 632.9,147.2 633.0,146.4 633.1,145.6 633.1,144.8 633.1,144.0 633.1,143.1 633.0,142.3 632.9,141.4 632.8,140.5 632.7,139.7 632.5,138.8 632.4,137.9 632.2,137.0 631.9,136.1 631.7,135.1 631.4,134.2 631.1,133.3 630.7,132.4 630.3,131.5 630.0,130.5 629.5,129.6 629.1,128.7 628.6,127.7 628.2,126.8 627.6,125.9 627.1,124.9 626.6,124.0 626.0,123.1 625.4,122.1 624.8,121.2 624.1,120.3 623.5,119.4 622.8,118.5 622.1,117.6 621.4,116.7 620.6,115.8 619.9,115.0 619.1,114.1 618.3,113.2 617.5,112.4 616.6,111.6 615.8,110.7 614.9,109.9 614.1,109.1 613.2,108.3 612.3,107.6 611.4,106.8" stroke-dasharray="4 3"/><polyline class="dg-line" points="626.5,88.3 624.8,86.9 623.0,85.6 621.3,84.2 619.5,82.9 617.7,81.7 615.9,80.4 614.0,79.2 612.2,78.0 610.3,76.9 608.5,75.8 606.6,74.7 604.7,73.7 602.8,72.7 600.9,71.7 599.0,70.8 597.1,69.9 595.1,69.1 593.2,68.3 591.3,67.5 589.4,66.8 587.5,66.1 585.6,65.5 583.7,64.9 581.8,64.3 579.9,63.8 578.0,63.4 576.2,63.0 574.3,62.6 572.5,62.3 570.7,62.0 568.9,61.7 567.1,61.5 565.3,61.4 563.5,61.3 561.8,61.2 560.1,61.2 558.4,61.3 556.8,61.4 555.1,61.5 553.5,61.7 551.9,61.9 550.4,62.1 548.9,62.4 547.4,62.8 545.9,63.2 544.5,63.6 543.1,64.1 541.7,64.7 540.4,65.2 539.1,65.9 537.9,66.5 536.7,67.2 535.5,68.0 534.4,68.7 533.3,69.6 532.2,70.4 531.2,71.3 530.3,72.3 529.4,73.3 528.5,74.3 527.7,75.3 526.9,76.4 526.1,77.6 525.4,78.7 524.8,79.9 524.2,81.2 523.7,82.4 523.2,83.7 522.7,85.0 522.3,86.4 521.9,87.8 521.6,89.2 521.4,90.6 521.2,92.1 521.0,93.6 520.9,95.1 520.9,96.6 520.9,98.2 520.9,99.7 521.0,101.3 521.1,102.9 521.3,104.6 521.6,106.2 521.9,107.9 522.2,109.5 522.6,111.2 523.1,112.9 523.6,114.6 524.1,116.3 524.7,118.1 525.3,119.8 526.0,121.5 526.7,123.3 527.5,125.0 528.3,126.8 529.2,128.5 530.1,130.3 531.0,132.0 532.0,133.8 533.1,135.5 534.2,137.2 535.3,139.0 536.4,140.7 537.6,142.4 538.9,144.1 540.2,145.8 541.5,147.5 542.8,149.1 544.2,150.8 545.6,152.4 547.1,154.0 548.6,155.6 550.1,157.2 551.6,158.8 553.2,160.3 554.8,161.8 556.4,163.3 558.1,164.8 559.8,166.2 561.5,167.7 563.2,169.1 565.0,170.4 566.7,171.8 568.5,173.1 570.3,174.3 572.1,175.6 574.0,176.8 575.8,178.0 577.7,179.1 579.5,180.2 581.4,181.3 583.3,182.3 585.2,183.3 587.1,184.3 589.0,185.2 590.9,186.1 592.9,186.9 594.8,187.7 596.7,188.5 598.6,189.2 600.5,189.9 602.4,190.5 604.3,191.1 606.2,191.7 608.1,192.2 610.0,192.6 611.8,193.0 613.7,193.4 615.5,193.7 617.3,194.0 619.1,194.3 620.9,194.5 622.7,194.6 624.5,194.7 626.2,194.8 627.9,194.8 629.6,194.7 631.2,194.6 632.9,194.5 634.5,194.3 636.1,194.1 637.6,193.9 639.1,193.6 640.6,193.2 642.1,192.8 643.5,192.4 644.9,191.9 646.3,191.3 647.6,190.8 648.9,190.1 650.1,189.5 651.3,188.8 652.5,188.0 653.6,187.3 654.7,186.4 655.8,185.6 656.8,184.7 657.7,183.7 658.6,182.7 659.5,181.7 660.3,180.7 661.1,179.6 661.9,178.4 662.6,177.3 663.2,176.1 663.8,174.8 664.3,173.6 664.8,172.3 665.3,171.0 665.7,169.6 666.1,168.2 666.4,166.8 666.6,165.4 666.8,163.9 667.0,162.4 667.1,160.9 667.1,159.4 667.1,157.8 667.1,156.3 667.0,154.7 666.9,153.1 666.7,151.4 666.4,149.8 666.1,148.1 665.8,146.5 665.4,144.8 664.9,143.1 664.4,141.4 663.9,139.7 663.3,137.9 662.7,136.2 662.0,134.5 661.3,132.7 660.5,131.0 659.7,129.2 658.8,127.5 657.9,125.7 657.0,124.0 656.0,122.2 654.9,120.5 653.8,118.8 652.7,117.0 651.6,115.3 650.4,113.6 649.1,111.9 647.8,110.2 646.5,108.5 645.2,106.9 643.8,105.2 642.4,103.6 640.9,102.0 639.4,100.4 637.9,98.8 636.4,97.2 634.8,95.7 633.2,94.2 631.6,92.7 629.9,91.2 628.2,89.8 626.5,88.3" stroke-dasharray="4 3"/><polyline class="dg-line" points="641.6,69.9 639.1,67.9 636.5,65.9 633.9,63.9 631.3,62.0 628.7,60.2 626.0,58.4 623.3,56.6 620.6,54.9 617.9,53.2 615.2,51.6 612.4,50.0 609.6,48.5 606.9,47.0 604.1,45.6 601.3,44.3 598.5,43.0 595.7,41.7 592.9,40.6 590.1,39.5 587.3,38.4 584.5,37.4 581.7,36.5 578.9,35.6 576.1,34.8 573.4,34.1 570.6,33.4 567.9,32.8 565.2,32.2 562.5,31.8 559.8,31.3 557.2,31.0 554.6,30.7 552.0,30.5 549.4,30.4 546.9,30.3 544.4,30.3 541.9,30.3 539.5,30.4 537.1,30.6 534.7,30.9 532.4,31.2 530.2,31.6 527.9,32.0 525.7,32.6 523.6,33.1 521.5,33.8 519.5,34.5 517.5,35.3 515.6,36.1 513.7,37.0 511.9,38.0 510.1,39.0 508.4,40.1 506.7,41.3 505.1,42.5 503.6,43.7 502.1,45.1 500.7,46.4 499.4,47.9 498.1,49.4 496.9,50.9 495.7,52.5 494.7,54.2 493.6,55.9 492.7,57.6 491.8,59.4 491.0,61.3 490.3,63.2 489.6,65.1 489.0,67.1 488.5,69.1 488.1,71.2 487.7,73.3 487.4,75.4 487.2,77.6 487.0,79.8 486.9,82.0 486.9,84.3 487.0,86.6 487.1,88.9 487.4,91.3 487.6,93.7 488.0,96.1 488.4,98.5 488.9,101.0 489.5,103.4 490.2,105.9 490.9,108.4 491.7,110.9 492.5,113.5 493.5,116.0 494.5,118.5 495.5,121.1 496.7,123.6 497.9,126.2 499.1,128.8 500.5,131.3 501.8,133.9 503.3,136.4 504.8,139.0 506.4,141.5 508.0,144.0 509.7,146.6 511.5,149.1 513.3,151.6 515.2,154.0 517.1,156.5 519.1,158.9 521.1,161.3 523.2,163.7 525.3,166.1 527.5,168.4 529.7,170.8 532.0,173.0 534.3,175.3 536.6,177.5 539.0,179.7 541.4,181.9 543.9,184.0 546.4,186.1 548.9,188.1 551.5,190.1 554.1,192.1 556.7,194.0 559.3,195.8 562.0,197.6 564.7,199.4 567.4,201.1 570.1,202.8 572.8,204.4 575.6,206.0 578.4,207.5 581.1,209.0 583.9,210.4 586.7,211.7 589.5,213.0 592.3,214.3 595.1,215.4 597.9,216.5 600.7,217.6 603.5,218.6 606.3,219.5 609.1,220.4 611.9,221.2 614.6,221.9 617.4,222.6 620.1,223.2 622.8,223.8 625.5,224.2 628.2,224.7 630.8,225.0 633.4,225.3 636.0,225.5 638.6,225.6 641.1,225.7 643.6,225.7 646.1,225.7 648.5,225.6 650.9,225.4 653.3,225.1 655.6,224.8 657.8,224.4 660.1,224.0 662.3,223.4 664.4,222.9 666.5,222.2 668.5,221.5 670.5,220.7 672.4,219.9 674.3,219.0 676.1,218.0 677.9,217.0 679.6,215.9 681.3,214.7 682.9,213.5 684.4,212.3 685.9,210.9 687.3,209.6 688.6,208.1 689.9,206.6 691.1,205.1 692.3,203.5 693.3,201.8 694.4,200.1 695.3,198.4 696.2,196.6 697.0,194.7 697.7,192.8 698.4,190.9 699.0,188.9 699.5,186.9 699.9,184.8 700.3,182.7 700.6,180.6 700.8,178.4 701.0,176.2 701.1,174.0 701.1,171.7 701.0,169.4 700.9,167.1 700.6,164.7 700.4,162.3 700.0,159.9 699.6,157.5 699.1,155.0 698.5,152.6 697.8,150.1 697.1,147.6 696.3,145.1 695.5,142.5 694.5,140.0 693.5,137.5 692.5,134.9 691.3,132.4 690.1,129.8 688.9,127.2 687.5,124.7 686.2,122.1 684.7,119.6 683.2,117.0 681.6,114.5 680.0,112.0 678.3,109.4 676.5,106.9 674.7,104.4 672.8,102.0 670.9,99.5 668.9,97.1 666.9,94.7 664.8,92.3 662.7,89.9 660.5,87.6 658.3,85.2 656.0,83.0 653.7,80.7 651.4,78.5 649.0,76.3 646.6,74.1 644.1,72.0 641.6,69.9" stroke-dasharray="4 3"/><polyline class="dg-line-accent" points="653.4,55.5 650.3,53.0 647.1,50.5 643.8,48.0 640.6,45.7 637.3,43.3 634.0,41.1 630.6,38.9 627.2,36.7 623.8,34.6 620.4,32.6 617.0,30.6 613.5,28.7 610.0,26.9 606.6,25.2 603.1,23.5 599.6,21.9 596.1,20.3 592.6,18.9 589.1,17.5 585.6,16.2 582.1,14.9 578.6,13.8 575.2,12.7 571.7,11.7 568.3,10.8 564.8,9.9 561.4,9.2 558.1,8.5 554.7,7.9 551.4,7.4 548.1,6.9 544.8,6.6 541.6,6.3 538.4,6.1 535.2,6.0 532.1,6.0 529.0,6.1 526.0,6.2 523.0,6.5 520.0,6.8 517.1,7.2 514.3,7.7 511.5,8.2 508.8,8.9 506.1,9.6 503.5,10.4 501.0,11.3 498.5,12.3 496.1,13.3 493.7,14.4 491.5,15.7 489.3,16.9 487.1,18.3 485.1,19.7 483.1,21.2 481.2,22.8 479.3,24.5 477.6,26.2 475.9,28.0 474.3,29.9 472.8,31.8 471.3,33.8 470.0,35.9 468.7,38.0 467.6,40.2 466.5,42.4 465.5,44.7 464.6,47.1 463.7,49.5 463.0,52.0 462.3,54.5 461.8,57.1 461.3,59.7 461.0,62.4 460.7,65.1 460.5,67.8 460.4,70.6 460.4,73.5 460.4,76.3 460.6,79.2 460.9,82.2 461.2,85.2 461.7,88.2 462.2,91.2 462.9,94.3 463.6,97.3 464.4,100.4 465.3,103.6 466.3,106.7 467.3,109.9 468.5,113.0 469.7,116.2 471.1,119.4 472.5,122.6 474.0,125.8 475.6,128.9 477.2,132.1 479.0,135.3 480.8,138.5 482.7,141.7 484.7,144.9 486.7,148.0 488.8,151.2 491.0,154.3 493.3,157.4 495.6,160.5 498.0,163.6 500.5,166.6 503.0,169.6 505.6,172.6 508.3,175.6 511.0,178.5 513.8,181.4 516.6,184.2 519.5,187.0 522.4,189.8 525.4,192.6 528.4,195.2 531.5,197.9 534.6,200.5 537.7,203.0 540.9,205.5 544.2,208.0 547.4,210.3 550.7,212.7 554.0,214.9 557.4,217.1 560.8,219.3 564.2,221.4 567.6,223.4 571.0,225.4 574.5,227.3 578.0,229.1 581.4,230.8 584.9,232.5 588.4,234.1 591.9,235.7 595.4,237.1 598.9,238.5 602.4,239.8 605.9,241.1 609.4,242.2 612.8,243.3 616.3,244.3 619.7,245.2 623.2,246.1 626.6,246.8 629.9,247.5 633.3,248.1 636.6,248.6 639.9,249.1 643.2,249.4 646.4,249.7 649.6,249.9 652.8,250.0 655.9,250.0 659.0,249.9 662.0,249.8 665.0,249.5 668.0,249.2 670.9,248.8 673.7,248.3 676.5,247.8 679.2,247.1 681.9,246.4 684.5,245.6 687.0,244.7 689.5,243.7 691.9,242.7 694.3,241.6 696.5,240.3 698.7,239.1 700.9,237.7 702.9,236.3 704.9,234.8 706.8,233.2 708.7,231.5 710.4,229.8 712.1,228.0 713.7,226.1 715.2,224.2 716.7,222.2 718.0,220.1 719.3,218.0 720.4,215.8 721.5,213.6 722.5,211.3 723.4,208.9 724.3,206.5 725.0,204.0 725.7,201.5 726.2,198.9 726.7,196.3 727.0,193.6 727.3,190.9 727.5,188.2 727.6,185.4 727.6,182.5 727.6,179.7 727.4,176.8 727.1,173.8 726.8,170.8 726.3,167.8 725.8,164.8 725.1,161.7 724.4,158.7 723.6,155.6 722.7,152.4 721.7,149.3 720.7,146.1 719.5,143.0 718.3,139.8 716.9,136.6 715.5,133.4 714.0,130.2 712.4,127.1 710.8,123.9 709.0,120.7 707.2,117.5 705.3,114.3 703.3,111.1 701.3,108.0 699.2,104.8 697.0,101.7 694.7,98.6 692.4,95.5 690.0,92.4 687.5,89.4 685.0,86.4 682.4,83.4 679.7,80.4 677.0,77.5 674.2,74.6 671.4,71.8 668.5,69.0 665.6,66.2 662.6,63.4 659.6,60.8 656.5,58.1 653.4,55.5"/></g><text class="dg-sub" x="709.0" y="254.0" text-anchor="end">β₁</text><text class="dg-sub" x="528.5" y="56.0">β₂</text><circle class="dg-dot" cx="594.0" cy="128.0" r="4"/><text class="dg-strong" x="602.0" y="132.0">β̂</text><circle class="dg-pt-a" cx="553.3" cy="214.4" r="5"/><text class="dg-key" x="448" y="202.44269374060497">both non-zero</text><text class="dg-sub" x="380.0" y="386" text-anchor="middle">β̂: least squares;  dashed: RSS contours;  solid: the first contour to touch the shaded budget</text></svg>
</div>
<figcaption>Constraint geometry of the lasso (left) and ridge (right), after ISLR Figure 6.7. The contours and touching points are computed, not sketched.</figcaption>
</figure>

**Entropy regularization** (reference 20) adds a bonus for the entropy of the model's predicted distribution, discouraging overconfident outputs.

- In **reinforcement learning** (A3C, PPO, soft actor-critic), an entropy bonus on the policy keeps it exploring instead of collapsing early onto one action.
- In classification, a **confidence penalty** (subtracting β times the output entropy from the loss) works much like label smoothing: it stops the model from becoming overconfident and improves calibration.
- **Entropy minimization**, the opposite sign, is used in semi-supervised learning to push predictions on unlabeled data toward confident decisions.

**Dropout** (reference 22) randomly switches off each unit of a neural network with probability p during training, a fresh random choice for every example. No unit can rely on a specific other unit being present, which prevents **co-adaptation**, where units only work in fixed combinations. Training with dropout is like training an exponentially large ensemble of thinned networks that share weights; at test time all units are kept and outputs are rescaled so their expected value matches training. Typical rates are 0.5 for large fully connected layers and 0.1 to 0.3 elsewhere, including transformers.

**Other regularizers to name:** **early stopping** (stop training when validation loss stops improving), **data augmentation** (flips, crops, noise, paraphrases that create plausible new training examples), **label smoothing** (Chapter 8), **batch normalization** (which adds a mild noise effect), smaller models, and more data, the most reliable regularizer of all.

**Validation: estimating performance honestly.** You cannot judge a model by its training error, so you hold data out. A single train/validation split is simple but noisy when data is small, and wastes data.

**k-fold cross-validation** (reference 21) splits the data into k equal folds. Each fold takes one turn as the validation set while the model trains on the other k − 1; the k scores are averaged, and their spread shows how stable the estimate is. Every example is used for validation exactly once and for training k − 1 times.

<figure class="diagram">
<div class="diagram__scroll">
<svg viewBox="0 0 760 290" style="--dg-w: 760" role="img" aria-labelledby="ch9-cv-title ch9-cv-desc"><title id="ch9-cv-title">Five-fold cross-validation</title><desc id="ch9-cv-desc">The data is split into five folds. In round 1 fold 1 is the validation set and folds 2 to 5 train the model; in round 2 fold 2 validates; and so on until round 5. Each round produces a score, and the five scores are averaged into the cross-validation estimate, with their spread showing how stable it is.</desc><defs><marker id="ch9-cv-arrow" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="dg-arrowhead" d="M0,0 L8,4 L0,8 z"/></marker></defs><text class="dg-sub" x="130.0" y="32" text-anchor="middle">fold 1</text><text class="dg-sub" x="210.0" y="32" text-anchor="middle">fold 2</text><text class="dg-sub" x="290.0" y="32" text-anchor="middle">fold 3</text><text class="dg-sub" x="370.0" y="32" text-anchor="middle">fold 4</text><text class="dg-sub" x="450.0" y="32" text-anchor="middle">fold 5</text><text class="dg-strong" x="10" y="69.0">round 1</text><rect class="dg-box dg-box--accent" x="92" y="47" width="76" height="34"/><text class="dg-head-text" x="130.0" y="69.0" text-anchor="middle">validate</text><rect class="dg-box" x="172" y="47" width="76" height="34"/><text class="dg-text" x="210.0" y="69.0" text-anchor="middle">train</text><rect class="dg-box" x="252" y="47" width="76" height="34"/><text class="dg-text" x="290.0" y="69.0" text-anchor="middle">train</text><rect class="dg-box" x="332" y="47" width="76" height="34"/><text class="dg-text" x="370.0" y="69.0" text-anchor="middle">train</text><rect class="dg-box" x="412" y="47" width="76" height="34"/><text class="dg-text" x="450.0" y="69.0" text-anchor="middle">train</text><path class="dg-line" d="M496,64.0 H526" marker-end="url(#ch9-cv-arrow)"/><text class="dg-text" x="534" y="69.0">score 1</text><text class="dg-strong" x="10" y="109.0">round 2</text><rect class="dg-box" x="92" y="87" width="76" height="34"/><text class="dg-text" x="130.0" y="109.0" text-anchor="middle">train</text><rect class="dg-box dg-box--accent" x="172" y="87" width="76" height="34"/><text class="dg-head-text" x="210.0" y="109.0" text-anchor="middle">validate</text><rect class="dg-box" x="252" y="87" width="76" height="34"/><text class="dg-text" x="290.0" y="109.0" text-anchor="middle">train</text><rect class="dg-box" x="332" y="87" width="76" height="34"/><text class="dg-text" x="370.0" y="109.0" text-anchor="middle">train</text><rect class="dg-box" x="412" y="87" width="76" height="34"/><text class="dg-text" x="450.0" y="109.0" text-anchor="middle">train</text><path class="dg-line" d="M496,104.0 H526" marker-end="url(#ch9-cv-arrow)"/><text class="dg-text" x="534" y="109.0">score 2</text><text class="dg-strong" x="10" y="149.0">round 3</text><rect class="dg-box" x="92" y="127" width="76" height="34"/><text class="dg-text" x="130.0" y="149.0" text-anchor="middle">train</text><rect class="dg-box" x="172" y="127" width="76" height="34"/><text class="dg-text" x="210.0" y="149.0" text-anchor="middle">train</text><rect class="dg-box dg-box--accent" x="252" y="127" width="76" height="34"/><text class="dg-head-text" x="290.0" y="149.0" text-anchor="middle">validate</text><rect class="dg-box" x="332" y="127" width="76" height="34"/><text class="dg-text" x="370.0" y="149.0" text-anchor="middle">train</text><rect class="dg-box" x="412" y="127" width="76" height="34"/><text class="dg-text" x="450.0" y="149.0" text-anchor="middle">train</text><path class="dg-line" d="M496,144.0 H526" marker-end="url(#ch9-cv-arrow)"/><text class="dg-text" x="534" y="149.0">score 3</text><text class="dg-strong" x="10" y="189.0">round 4</text><rect class="dg-box" x="92" y="167" width="76" height="34"/><text class="dg-text" x="130.0" y="189.0" text-anchor="middle">train</text><rect class="dg-box" x="172" y="167" width="76" height="34"/><text class="dg-text" x="210.0" y="189.0" text-anchor="middle">train</text><rect class="dg-box" x="252" y="167" width="76" height="34"/><text class="dg-text" x="290.0" y="189.0" text-anchor="middle">train</text><rect class="dg-box dg-box--accent" x="332" y="167" width="76" height="34"/><text class="dg-head-text" x="370.0" y="189.0" text-anchor="middle">validate</text><rect class="dg-box" x="412" y="167" width="76" height="34"/><text class="dg-text" x="450.0" y="189.0" text-anchor="middle">train</text><path class="dg-line" d="M496,184.0 H526" marker-end="url(#ch9-cv-arrow)"/><text class="dg-text" x="534" y="189.0">score 4</text><text class="dg-strong" x="10" y="229.0">round 5</text><rect class="dg-box" x="92" y="207" width="76" height="34"/><text class="dg-text" x="130.0" y="229.0" text-anchor="middle">train</text><rect class="dg-box" x="172" y="207" width="76" height="34"/><text class="dg-text" x="210.0" y="229.0" text-anchor="middle">train</text><rect class="dg-box" x="252" y="207" width="76" height="34"/><text class="dg-text" x="290.0" y="229.0" text-anchor="middle">train</text><rect class="dg-box" x="332" y="207" width="76" height="34"/><text class="dg-text" x="370.0" y="229.0" text-anchor="middle">train</text><rect class="dg-box dg-box--accent" x="412" y="207" width="76" height="34"/><text class="dg-head-text" x="450.0" y="229.0" text-anchor="middle">validate</text><path class="dg-line" d="M496,224.0 H526" marker-end="url(#ch9-cv-arrow)"/><text class="dg-text" x="534" y="229.0">score 5</text><path class="dg-line" d="M602,64.0 H614 V224.0 H602"/><path class="dg-line" d="M614,144.0 H630" marker-end="url(#ch9-cv-arrow)"/><rect class="dg-shadow" x="636" y="120.0" width="100" height="56"/><rect class="dg-box dg-box--accent" x="632" y="116.0" width="100" height="56"/><text class="dg-head-text" x="682" y="142.0" text-anchor="middle">CV</text><text class="dg-head-text" x="682" y="159.0" text-anchor="middle">estimate</text><text class="dg-sub" x="290.0" y="274" text-anchor="middle">each example is validated once and used for training k − 1 = 4 times</text><text class="dg-sub" x="682" y="194.0" text-anchor="middle">mean of 5,</text><text class="dg-sub" x="682" y="210.0" text-anchor="middle">spread = stability</text></svg>
</div>
<figcaption>5-fold cross-validation: every fold takes one turn as the validation set, and the CV estimate is the average of the five scores.</figcaption>
</figure>

| Method                  | How                                                      | Strengths                                       | Weaknesses                                               |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| Hold-out split          | One train/validation split                               | Fast; fine with lots of data                    | Noisy estimate on small data; wastes data                |
| k-fold CV (k = 5 or 10) | k rotations of k − 1 train folds and 1 validation fold   | Uses all data; gives a spread                   | k times the training cost                                |
| Leave-one-out (LOOCV)   | k = n: each row is its own fold                          | Almost unbiased; deterministic                  | n fits; high-variance estimate; too slow for most models |
| Stratified k-fold       | k-fold that keeps class proportions in each fold         | Stable on imbalanced data                       | Classification only                                      |
| Group k-fold            | All rows of an entity stay in one fold                   | Prevents group leakage                          | Needs group labels; folds may be uneven                  |
| Time-series split       | Train on the past, validate on the next block            | Respects time order                             | Fewer, smaller training sets early on                    |
| Nested CV               | An inner CV tunes hyperparameters inside each outer fold | Unbiased estimate of the whole tuning procedure | k_outer × k_inner × grid size fits                       |

**The tuning trap.** If you try many hyperparameter settings and report the best cross-validation score, that score is optimistic: you selected it partly for its luck. Report performance from data that played no part in the selection, either a held-out test set or **nested cross-validation** (9.3 shows the bias on pure noise).

## 9.2 Core math

**Penalized objectives.** For a linear model with weights β and regularization strength λ:

$$
\text{Ridge: } \min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2, \qquad \text{Lasso: } \min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1, \qquad \text{Elastic net: } + \lambda\big(\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2\big)
$$

The intercept is not penalized, and features must be **standardized** first, because the penalty treats every coefficient alike and a feature in small units would otherwise need, and be punished for, a large coefficient.

**Ridge has a closed form.** Setting the gradient to zero adds λI to the normal equations (Chapter 2):

$$
\hat\beta_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

Adding λI makes the matrix invertible even with perfectly collinear features or more features than rows. In terms of the singular values d_j of X, ridge shrinks the fit along each principal direction by d_j² / (d_j² + λ): directions with little variance in the data are shrunk the most.

**Lasso has no closed form, but each coordinate does.** With standardized features, coordinate descent updates one coefficient at a time by **soft-thresholding** the least-squares value z_j:

$$
S(z, \lambda) = \operatorname{sign}(z)\,\max(|z| - \lambda,\, 0)
$$

Any coefficient whose least-squares value is smaller than the threshold becomes exactly zero; larger ones are pulled toward zero by λ. Compare ridge, which scales every coefficient by the same factor and never reaches zero. In the orthonormal-design case the two rules are, with λ′ the matching threshold:

$$
\hat\beta_j^{\text{ridge}} = \frac{\hat\beta_j^{\text{OLS}}}{1 + \lambda}, \qquad \hat\beta_j^{\text{lasso}} = S\big(\hat\beta_j^{\text{OLS}}, \lambda'\big)
$$

**The Bayesian view.** Regularized least squares is **maximum a posteriori (MAP)** estimation: the penalty is the negative log of a prior on the weights. A Gaussian prior gives ridge; a Laplace prior, sharply peaked at zero, gives the lasso.

$$
-\log p(\beta \mid y) = \underbrace{-\log p(y \mid \beta)}_{\text{squared error}} \underbrace{- \log p(\beta)}_{\text{penalty}} + \text{const}, \qquad \beta_j \sim \mathcal{N}(0, \tau^2) \Rightarrow \text{L2}, \quad \beta_j \sim \text{Laplace}(0, b) \Rightarrow \text{L1}
$$

**Weight decay versus L2 in adaptive optimizers.** With plain SGD, adding λ‖w‖²/2 to the loss and multiplying the weights by (1 − ηλ) each step are the same thing. With Adam they are not: an L2 term's gradient is divided by Adam's per-weight scaling, so weights with large gradients are barely regularized. **AdamW** applies the decay directly to the weights, outside the adaptive step, and is the standard for training transformers (Chapter 10):

$$
w \leftarrow w - \eta\left(\frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda\, w\right)
$$

**Entropy regularization.** For a predicted distribution π over actions or classes, subtract β times its entropy from the loss. In policy-gradient RL, with advantage A:

$$
H(\pi) = -\sum_a \pi(a)\log\pi(a), \qquad L = -\log\pi(a_t)\,A_t \;-\; \beta\, H\big(\pi(\cdot \mid s_t)\big)
$$

Entropy is largest (log K for K options) when the distribution is uniform, so the bonus pushes away from collapsing onto one choice. Example: logits (2.0, 0.5, −1.0) give probabilities (0.786, 0.175, 0.039) and entropy 0.622 nats, against a maximum of log 3 = 1.099.

**Inverted dropout.** During training, each activation h_i is kept with probability 1 − p and scaled up by 1/(1 − p); at test time nothing is dropped and nothing is scaled. The expected activation is unchanged:

$$
\tilde h_i = \frac{m_i\, h_i}{1 - p}, \quad m_i \sim \text{Bernoulli}(1 - p), \qquad \mathbb{E}[\tilde h_i] = h_i
$$

The original paper instead scaled weights by (1 − p) at test time; the inverted form moves that work to training so inference stays untouched. For a linear model with squared loss, dropout on the inputs is equivalent in expectation to an L2 penalty scaled by each feature's variance. Keeping dropout on at test time and averaging several stochastic passes, **Monte Carlo dropout**, gives a cheap estimate of prediction uncertainty.

**Cross-validation estimate.** With k folds F_1 … F_k and the model f̂^(−j) trained without fold j:

$$
\text{CV}_{(k)} = \frac{1}{k}\sum_{j=1}^{k} \text{Err}\big(\hat f^{(-j)},\ F_j\big)
$$

**Choosing k is a bias-variance trade-off** for the estimate itself. Small k trains on less data, so the estimate is pessimistic (biased upward). LOOCV (k = n) is nearly unbiased but averages n highly correlated models, so its variance is high. k = 5 or 10 is the usual compromise. For least-squares linear regression, LOOCV even has a shortcut needing one fit, where h_ii is the **leverage** of point i, the i-th diagonal of the hat matrix X(XᵀX)⁻¹Xᵀ:

$$
\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n} \left(\frac{y_i - \hat y_i}{1 - h_{ii}}\right)^2
$$

**The one-standard-error rule** (from ISLR): among models whose CV error is within one standard error of the best, choose the simplest. It guards against picking a complex model on noise.

## 9.3 Code

**Ridge, lasso and elastic net on two nearly identical features.** y depends on x₁ only (true coefficient 3); x₂ is x₁ plus tiny noise.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

rng = np.random.default_rng(0)
x1 = rng.normal(size=100)
x2 = x1 + rng.normal(scale=0.01, size=100)                  # almost a copy of x1
y = 3 * x1 + rng.normal(scale=0.5, size=100)
X = np.c_[x1, x2]

for name, m in [("OLS", LinearRegression()), ("Ridge", Ridge(alpha=1.0)),
                ("Lasso", Lasso(alpha=0.1)), ("Elastic net", ElasticNet(alpha=0.1, l1_ratio=0.5))]:
    print(name, m.fit(X, y).coef_.round(3))
# OLS          [1.156 1.835]   unstable split of the effect
# Ridge        [1.486 1.49 ]   shared evenly, sum close to 3
# Lasso        [2.884 0.   ]   one kept, one dropped
# Elastic net  [1.43  1.43 ]   shared, with some shrinkage

# Ridge closed form on centered data matches scikit-learn
Xc, yc = X - X.mean(0), y - y.mean()
print(np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(2), Xc.T @ yc).round(3))   # [1.486 1.49 ]
```

**Lasso as a feature selector.** 200 rows, 50 standardized features, 10 truly informative:

```python
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y, true_coef = make_regression(n_samples=200, n_features=50, n_informative=10,
                                  noise=5.0, coef=True, random_state=0)
X = StandardScaler().fit_transform(X)
for alpha in [0.01, 0.1, 1, 5, 20]:
    lasso = Lasso(alpha=alpha, max_iter=10_000).fit(X, y)
    ridge = Ridge(alpha=alpha).fit(X, y)
    print(alpha, (lasso.coef_ != 0).sum(), (ridge.coef_ != 0).sum())
# alpha    lasso non-zero   ridge non-zero
# 0.01           50               50
# 0.1            43               50
# 1              11               50     <- all 10 true features + 1 false positive
# 5              10               50
# 20              7               50     <- now dropping true features too
```

**Soft-thresholding and inverted dropout from scratch:**

```python
def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

print(soft_threshold(np.array([-3, -0.5, 0.2, 2.0]), 1.0))   # [-2. -0.  0.  1.]

def dropout(x, p, training=True, rng=np.random.default_rng(0)):
    if not training or p == 0:
        return x                                           # inference: identity
    mask = rng.random(x.shape) >= p                        # keep with probability 1 - p
    return x * mask / (1 - p)                              # rescale so E[output] = x

out = dropout(np.ones((1000, 1000)), p=0.5)
print(out.mean(), (out == 0).mean())                       # 0.9996 0.5002
```

**Why nested cross-validation matters: tuning on pure noise.** 100 rows, 200 random features, random labels: no model can beat 50% accuracy. Tune an RBF SVM over 25 (C, γ) settings, and repeat for 10 random label draws.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 200))
grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1]}
non_nested, nested = [], []
for seed in range(10):
    y = np.random.default_rng(seed).integers(0, 2, 100)     # labels are pure noise
    search = GridSearchCV(SVC(), grid, cv=KFold(5, shuffle=True, random_state=seed))
    non_nested.append(search.fit(X, y).best_score_)          # best inner CV score
    nested.append(cross_val_score(search, X, y,              # outer CV around the search
                                  cv=KFold(5, shuffle=True, random_state=seed + 100)).mean())
print(np.mean(non_nested), np.mean(nested))                  # 0.552 vs 0.484
```

The non-nested score claims 55% accuracy on data with no signal: picking the luckiest of 25 settings inflated it by about 5 points. Nested CV, which evaluates the entire tuning procedure on folds it never saw, reports 48%, within noise of the true 50%. The gap grows with the size of the search and shrinks with more data.

## 9.4 In system design

- **Sparse models for cheap serving.** L1-regularized logistic regression over millions of hashed features (common in ads and spam filtering) keeps only the useful weights, which shrinks the model and speeds up serving. The lasso also tells you which upstream features you could stop computing and logging.
- **Regularization strength is retuned when data changes.** More training data needs less regularization; a model retrained on a bigger window with the old λ is usually over-regularized. Include λ in the automated tuning of each retraining run.
- **Early stopping is regularization you get for free** in gradient-boosted trees and neural networks: monitor a validation metric each round or epoch and keep the best checkpoint.
- **Transformers** typically combine AdamW weight decay (often around 0.1, not applied to biases and normalization weights), dropout of 0.1 or less, and label smoothing; very large pretraining runs often use little or no dropout because they see each example only about once.
- **Validation strategy must mirror production** (Chapter 5): time-based folds for forecasting, group folds for repeated users, and a final untouched test set. For expensive deep models, one well-chosen validation split usually replaces k-fold CV.
- **Guard against tuning overfit.** Every time a team checks the test set, it becomes a little less trustworthy. Keep a frozen test set, log how often it is used, and refresh it periodically.
- **Uncertainty with MC dropout** can flag low-confidence predictions for human review or abstention, with no extra models to train.

## 9.5 Interview Q&A

**Q1. What is the difference between L1 and L2 regularization?** L1 adds the sum of absolute weights; it drives many weights to exactly zero, giving sparse models and feature selection, and keeps one of a group of correlated features. L2 adds the sum of squared weights; it shrinks all weights smoothly, never exactly to zero, and spreads weight across correlated features. In Bayesian terms, L1 is a Laplace prior and L2 a Gaussian prior.

**Q2. Why does L1 produce sparse solutions?** Geometrically, the L1 constraint region is a diamond with corners on the axes, and the loss contours usually first touch it at a corner, where some coefficients are zero. Algebraically, the lasso's coordinate update is soft-thresholding, which sets any coefficient whose least-squares value is below λ exactly to zero; L2's update only rescales.

**Q3. When would you choose elastic net?** When you want sparsity but have groups of correlated features: the lasso picks one arbitrarily and can be unstable, while the L2 part of elastic net keeps correlated features together. Also when there are more features than rows, where the lasso can select at most n features.

**Q4. Why must features be standardized before ridge or lasso?** The penalty applies the same λ to every coefficient. A feature measured in small units needs a large coefficient and gets penalized heavily, so the result depends on arbitrary units. Standardizing puts all features on the same scale.

**Q5. How does dropout work, and what happens at test time?** During training each unit is zeroed with probability p, and with inverted dropout the survivors are scaled by 1/(1 − p) so expected activations are unchanged. At test time no units are dropped and no scaling is needed. It prevents co-adaptation and acts like averaging an ensemble of thinned networks.

**Q6. What is the difference between L2 regularization and weight decay?** With plain SGD they are identical. With adaptive optimizers like Adam, an L2 term's gradient gets divided by the per-parameter adaptive scale, so heavily updated weights are under-regularized. AdamW decouples decay from the gradient step, which works better and is the standard for transformers.

**Q7. What is entropy regularization, and where is it used?** Adding a bonus for the entropy of the model's output distribution, which discourages overconfident, collapsed predictions. In reinforcement learning it keeps policies exploring (A3C, PPO, soft actor-critic); in classification it acts as a confidence penalty similar to label smoothing.

**Q8. How do you choose k in k-fold cross-validation?** Smaller k trains on less data, so the estimate is pessimistically biased; LOOCV is nearly unbiased but has high variance and costs n fits. k = 5 or 10 balances the two. Use stratified folds for imbalanced classes, group folds for repeated entities and time-ordered folds for temporal data.

**Q9. What is nested cross-validation, and why is it needed?** The inner loop tunes hyperparameters; the outer loop evaluates the whole tuning procedure on folds the inner loop never saw. Without it, the best score from a search is optimistically biased because it was selected partly for luck; on pure noise, a 25-setting search reported 55% accuracy while nested CV reported 48%.

**Q10. Your model's training loss keeps falling but validation loss started rising at epoch 12. What do you do?** The model is overfitting after epoch 12. Use early stopping (keep the epoch-12 checkpoint), and to train longer usefully add regularization: weight decay, dropout, augmentation or more data. Also check that the validation set is representative and not leaking.

## 9.6 Cheat sheet and pitfalls

**Remember:**

- Ridge: λ‖β‖₂², closed form (XᵀX + λI)⁻¹Xᵀy, Gaussian prior, shrinks but no zeros.
- Lasso: λ‖β‖₁, soft-thresholding, Laplace prior, exact zeros and feature selection.
- Elastic net: both; handles correlated groups and p > n.
- Standardize features; never penalize the intercept.
- AdamW: decoupled weight decay; plain L2 inside Adam under-regularizes.
- Entropy bonus: −βH(π); keeps RL policies exploring, curbs overconfidence.
- Inverted dropout: mask with keep-probability 1 − p, scale by 1/(1 − p), identity at test time.
- k-fold: k = 5 or 10; LOOCV nearly unbiased but high variance; nested CV for honest tuned scores; the one-standard-error rule picks simpler models.

**Pitfalls that cost candidates:**

- Saying L2 performs feature selection, or that L1 "shrinks less".
- Regularizing unscaled features, or fitting the scaler on all data before CV.
- Forgetting to turn dropout off at inference (call `model.eval()` in PyTorch).
- Reporting the best grid-search CV score as the expected test performance.
- Using random k-fold on time series or grouped data.
- Keeping the same λ after the training set grows several times larger.

## 9.7 Further reading

<ol start="19">
  <li><a href="https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning">L2 and L1 regularization in machine learning</a> (Analytics Steps)</li>
  <li><a href="https://paperswithcode.com/method/entropy-regularization">Entropy regularization</a> (Papers with Code)</li>
  <li><a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)">Cross-validation (statistics)</a> (Wikipedia)</li>
  <li>Srivastava et al., <a href="https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a> (JMLR 2014)</li>
</ol>

**Supplementary:** ISLR Chapter 5 (resampling methods: cross-validation and the bootstrap) and Chapter 6 (ridge, lasso and model selection); Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW).
