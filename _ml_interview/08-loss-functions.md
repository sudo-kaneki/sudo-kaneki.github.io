---
title: "Loss Functions"
chapter: 8
permalink: "/blog/cracking-the-ml-interview/loss-functions/"
updated: 2026-11-01
description: 'A loss function turns "how wrong was this prediction?" into one number that training can minimize.'
---

A loss function turns "how wrong was this prediction?" into one number that training can minimize. The choice is not cosmetic: it decides what the model's prediction means (a mean, a median, a probability), how much outliers matter, and how well training behaves. Interviewers use losses to check whether you can connect math, statistics and engineering in one answer.

## 8.1 Intuition

**A loss is a statement of what you care about.** Squared error says big misses are much worse than small ones. Absolute error says every unit of error costs the same. Cross-entropy says confident wrong answers are catastrophic. Each choice also implies a **noise model**: minimizing a loss is usually the same as maximizing the likelihood of the data under some assumed distribution (8.2).

**Loss, cost and metric.** The **loss** is the error on one example; the **cost** or **objective** is the average over the data, often plus a regularization term (Chapter 9). The **evaluation metric** (accuracy, F1, MAPE) is what the business reads; it is often not differentiable, so you train on a smooth **surrogate loss** that tracks it.

**Classification: cross-entropy** (reference 15). The model outputs a probability for each class, and cross-entropy charges −log(the probability given to the true class). Giving the true class 0.9 costs 0.105; giving it 0.01 costs 4.6. It is the default because it is the negative log-likelihood of a categorical (or, for two classes, Bernoulli) model, its gradient with a softmax or sigmoid is simply "prediction minus target", and it rewards well-calibrated probabilities.

**Regression: MSE, MAE and Huber** (references 16 to 18). Here r = y − ŷ is the **residual**, the error on one example.

- **MSE (mean squared error)** averages r². Squaring punishes large errors heavily, so a few outliers can dominate training. The best constant prediction under MSE is the **mean**. Its square root, **RMSE (root mean squared error)**, is in the target's units.
- **MAE (mean absolute error)** averages |r|. Every unit of error costs the same, so it is robust to outliers. The best constant prediction is the **median**. Its gradient has constant size (±1), so it does not shrink near the optimum, and it is not differentiable at 0.
- **Huber loss** is quadratic for small residuals (|r| ≤ δ) and linear beyond. It behaves like MSE near the optimum (smooth, shrinking gradients) and like MAE for outliers (bounded gradients). δ sets where "small" ends and is usually chosen relative to the noise scale.

<figure class="diagram">
<div class="diagram__scroll">
<svg viewBox="0 0 760 380" style="--dg-w: 760" role="img" aria-labelledby="ch8-loss-title ch8-loss-desc"><title id="ch8-loss-title">MSE, MAE and Huber loss and their gradients</title><desc id="ch8-loss-desc">Left: loss against the residual r from −3 to 3. MSE is a parabola reaching 9 at r = ±3; MAE is a V reaching 3; Huber with delta 1 follows the parabola for |r| ≤ 1 and grows linearly beyond, reaching 2.5. Right: gradients. MSE&#x27;s gradient is 2r, growing without bound; MAE&#x27;s jumps from −1 to +1 at zero; Huber&#x27;s equals r inside the shaded band and is capped at ±1 outside it.</desc><rect class="dg-shade" x="173.3" y="40" width="93.3" height="250"/><text class="dg-strong" x="220.0" y="24" text-anchor="middle">loss</text><text class="dg-sub" x="220.0" y="54" text-anchor="middle">|r| ≤ δ</text><rect class="dg-shade" x="560.0" y="40" width="90.0" height="250"/><text class="dg-strong" x="605.0" y="24" text-anchor="middle">gradient with respect to r</text><text class="dg-sub" x="605.0" y="54" text-anchor="middle">|r| ≤ δ</text><line class="dg-line" x1="80.0" y1="290.0" x2="360.0" y2="290.0"/><line class="dg-line" x1="80.0" y1="290.0" x2="80.0" y2="40.0"/><line class="dg-line" x1="80.0" y1="290.0" x2="80.0" y2="295.0"/><text class="dg-sub" x="80.0" y="310" text-anchor="middle">−3</text><line class="dg-line" x1="126.7" y1="290.0" x2="126.7" y2="295.0"/><text class="dg-sub" x="126.66666666666666" y="310" text-anchor="middle">−2</text><line class="dg-line" x1="173.3" y1="290.0" x2="173.3" y2="295.0"/><text class="dg-sub" x="173.33333333333331" y="310" text-anchor="middle">−1</text><line class="dg-line" x1="220.0" y1="290.0" x2="220.0" y2="295.0"/><text class="dg-sub" x="220.0" y="310" text-anchor="middle">0</text><line class="dg-line" x1="266.7" y1="290.0" x2="266.7" y2="295.0"/><text class="dg-sub" x="266.66666666666663" y="310" text-anchor="middle">1</text><line class="dg-line" x1="313.3" y1="290.0" x2="313.3" y2="295.0"/><text class="dg-sub" x="313.33333333333337" y="310" text-anchor="middle">2</text><line class="dg-line" x1="360.0" y1="290.0" x2="360.0" y2="295.0"/><text class="dg-sub" x="360.0" y="310" text-anchor="middle">3</text><line class="dg-line" x1="75.0" y1="290.0" x2="80.0" y2="290.0"/><text class="dg-sub" x="71" y="294.0" text-anchor="end">0</text><line class="dg-line" x1="75.0" y1="206.7" x2="80.0" y2="206.7"/><text class="dg-sub" x="71" y="210.66666666666669" text-anchor="end">3</text><line class="dg-line" x1="75.0" y1="123.3" x2="80.0" y2="123.3"/><text class="dg-sub" x="71" y="127.33333333333334" text-anchor="end">6</text><line class="dg-line" x1="75.0" y1="40.0" x2="80.0" y2="40.0"/><text class="dg-sub" x="71" y="44.0" text-anchor="end">9</text><text class="dg-sub" x="220.0" y="330" text-anchor="middle">residual r = y − ŷ</text><text class="dg-sub" x="0" y="0" text-anchor="middle" transform="translate(36,165.0) rotate(-90)">loss</text><line class="dg-line" x1="470.0" y1="290.0" x2="740.0" y2="290.0"/><line class="dg-line" x1="470.0" y1="290.0" x2="470.0" y2="40.0"/><line class="dg-line" x1="470.0" y1="290.0" x2="470.0" y2="295.0"/><text class="dg-sub" x="470.0" y="310" text-anchor="middle">−3</text><line class="dg-line" x1="515.0" y1="290.0" x2="515.0" y2="295.0"/><text class="dg-sub" x="515.0" y="310" text-anchor="middle">−2</text><line class="dg-line" x1="560.0" y1="290.0" x2="560.0" y2="295.0"/><text class="dg-sub" x="560.0" y="310" text-anchor="middle">−1</text><line class="dg-line" x1="605.0" y1="290.0" x2="605.0" y2="295.0"/><text class="dg-sub" x="605.0" y="310" text-anchor="middle">0</text><line class="dg-line" x1="650.0" y1="290.0" x2="650.0" y2="295.0"/><text class="dg-sub" x="650.0" y="310" text-anchor="middle">1</text><line class="dg-line" x1="695.0" y1="290.0" x2="695.0" y2="295.0"/><text class="dg-sub" x="695.0" y="310" text-anchor="middle">2</text><line class="dg-line" x1="740.0" y1="290.0" x2="740.0" y2="295.0"/><text class="dg-sub" x="740.0" y="310" text-anchor="middle">3</text><line class="dg-line" x1="465.0" y1="290.0" x2="470.0" y2="290.0"/><text class="dg-sub" x="461" y="294.0" text-anchor="end">−6</text><line class="dg-line" x1="465.0" y1="227.5" x2="470.0" y2="227.5"/><text class="dg-sub" x="461" y="231.5" text-anchor="end">−3</text><line class="dg-line" x1="465.0" y1="165.0" x2="470.0" y2="165.0"/><text class="dg-sub" x="461" y="169.0" text-anchor="end">0</text><line class="dg-line" x1="465.0" y1="102.5" x2="470.0" y2="102.5"/><text class="dg-sub" x="461" y="106.5" text-anchor="end">3</text><line class="dg-line" x1="465.0" y1="40.0" x2="470.0" y2="40.0"/><text class="dg-sub" x="461" y="44.0" text-anchor="end">6</text><text class="dg-sub" x="605.0" y="330" text-anchor="middle">residual r = y − ŷ</text><text class="dg-sub" x="0" y="0" text-anchor="middle" transform="translate(426,165.0) rotate(-90)">gradient</text><line class="dg-rule" x1="470.0" y1="165.0" x2="740.0" y2="165.0"/><polyline class="dg-line" points="80.0,206.7 80.9,207.2 81.9,207.8 82.8,208.3 83.7,208.9 84.7,209.4 85.6,210.0 86.5,210.6 87.5,211.1 88.4,211.7 89.3,212.2 90.3,212.8 91.2,213.3 92.1,213.9 93.1,214.4 94.0,215.0 94.9,215.6 95.9,216.1 96.8,216.7 97.7,217.2 98.7,217.8 99.6,218.3 100.5,218.9 101.5,219.4 102.4,220.0 103.3,220.6 104.3,221.1 105.2,221.7 106.1,222.2 107.1,222.8 108.0,223.3 108.9,223.9 109.9,224.4 110.8,225.0 111.7,225.6 112.7,226.1 113.6,226.7 114.5,227.2 115.5,227.8 116.4,228.3 117.3,228.9 118.3,229.4 119.2,230.0 120.1,230.6 121.1,231.1 122.0,231.7 122.9,232.2 123.9,232.8 124.8,233.3 125.7,233.9 126.7,234.4 127.6,235.0 128.5,235.6 129.5,236.1 130.4,236.7 131.3,237.2 132.3,237.8 133.2,238.3 134.1,238.9 135.1,239.4 136.0,240.0 136.9,240.6 137.9,241.1 138.8,241.7 139.7,242.2 140.7,242.8 141.6,243.3 142.5,243.9 143.5,244.4 144.4,245.0 145.3,245.6 146.3,246.1 147.2,246.7 148.1,247.2 149.1,247.8 150.0,248.3 150.9,248.9 151.9,249.4 152.8,250.0 153.7,250.6 154.7,251.1 155.6,251.7 156.5,252.2 157.5,252.8 158.4,253.3 159.3,253.9 160.3,254.4 161.2,255.0 162.1,255.6 163.1,256.1 164.0,256.7 164.9,257.2 165.9,257.8 166.8,258.3 167.7,258.9 168.7,259.4 169.6,260.0 170.5,260.6 171.5,261.1 172.4,261.7 173.3,262.2 174.3,262.8 175.2,263.3 176.1,263.9 177.1,264.4 178.0,265.0 178.9,265.6 179.9,266.1 180.8,266.7 181.7,267.2 182.7,267.8 183.6,268.3 184.5,268.9 185.5,269.4 186.4,270.0 187.3,270.6 188.3,271.1 189.2,271.7 190.1,272.2 191.1,272.8 192.0,273.3 192.9,273.9 193.9,274.4 194.8,275.0 195.7,275.6 196.7,276.1 197.6,276.7 198.5,277.2 199.5,277.8 200.4,278.3 201.3,278.9 202.3,279.4 203.2,280.0 204.1,280.6 205.1,281.1 206.0,281.7 206.9,282.2 207.9,282.8 208.8,283.3 209.7,283.9 210.7,284.4 211.6,285.0 212.5,285.6 213.5,286.1 214.4,286.7 215.3,287.2 216.3,287.8 217.2,288.3 218.1,288.9 219.1,289.4 220.0,290.0 220.9,289.4 221.9,288.9 222.8,288.3 223.7,287.8 224.7,287.2 225.6,286.7 226.5,286.1 227.5,285.6 228.4,285.0 229.3,284.4 230.3,283.9 231.2,283.3 232.1,282.8 233.1,282.2 234.0,281.7 234.9,281.1 235.9,280.6 236.8,280.0 237.7,279.4 238.7,278.9 239.6,278.3 240.5,277.8 241.5,277.2 242.4,276.7 243.3,276.1 244.3,275.6 245.2,275.0 246.1,274.4 247.1,273.9 248.0,273.3 248.9,272.8 249.9,272.2 250.8,271.7 251.7,271.1 252.7,270.6 253.6,270.0 254.5,269.4 255.5,268.9 256.4,268.3 257.3,267.8 258.3,267.2 259.2,266.7 260.1,266.1 261.1,265.6 262.0,265.0 262.9,264.4 263.9,263.9 264.8,263.3 265.7,262.8 266.7,262.2 267.6,261.7 268.5,261.1 269.5,260.6 270.4,260.0 271.3,259.4 272.3,258.9 273.2,258.3 274.1,257.8 275.1,257.2 276.0,256.7 276.9,256.1 277.9,255.6 278.8,255.0 279.7,254.4 280.7,253.9 281.6,253.3 282.5,252.8 283.5,252.2 284.4,251.7 285.3,251.1 286.3,250.6 287.2,250.0 288.1,249.4 289.1,248.9 290.0,248.3 290.9,247.8 291.9,247.2 292.8,246.7 293.7,246.1 294.7,245.6 295.6,245.0 296.5,244.4 297.5,243.9 298.4,243.3 299.3,242.8 300.3,242.2 301.2,241.7 302.1,241.1 303.1,240.6 304.0,240.0 304.9,239.4 305.9,238.9 306.8,238.3 307.7,237.8 308.7,237.2 309.6,236.7 310.5,236.1 311.5,235.6 312.4,235.0 313.3,234.4 314.3,233.9 315.2,233.3 316.1,232.8 317.1,232.2 318.0,231.7 318.9,231.1 319.9,230.6 320.8,230.0 321.7,229.4 322.7,228.9 323.6,228.3 324.5,227.8 325.5,227.2 326.4,226.7 327.3,226.1 328.3,225.6 329.2,225.0 330.1,224.4 331.1,223.9 332.0,223.3 332.9,222.8 333.9,222.2 334.8,221.7 335.7,221.1 336.7,220.6 337.6,220.0 338.5,219.4 339.5,218.9 340.4,218.3 341.3,217.8 342.3,217.2 343.2,216.7 344.1,216.1 345.1,215.6 346.0,215.0 346.9,214.4 347.9,213.9 348.8,213.3 349.7,212.8 350.7,212.2 351.6,211.7 352.5,211.1 353.5,210.6 354.4,210.0 355.3,209.4 356.3,208.9 357.2,208.3 358.1,207.8 359.1,207.2 360.0,206.7"/><polyline class="dg-line" points="80.0,220.6 80.9,221.1 81.9,221.7 82.8,222.2 83.7,222.8 84.7,223.3 85.6,223.9 86.5,224.4 87.5,225.0 88.4,225.6 89.3,226.1 90.3,226.7 91.2,227.2 92.1,227.8 93.1,228.3 94.0,228.9 94.9,229.4 95.9,230.0 96.8,230.6 97.7,231.1 98.7,231.7 99.6,232.2 100.5,232.8 101.5,233.3 102.4,233.9 103.3,234.4 104.3,235.0 105.2,235.6 106.1,236.1 107.1,236.7 108.0,237.2 108.9,237.8 109.9,238.3 110.8,238.9 111.7,239.4 112.7,240.0 113.6,240.6 114.5,241.1 115.5,241.7 116.4,242.2 117.3,242.8 118.3,243.3 119.2,243.9 120.1,244.4 121.1,245.0 122.0,245.6 122.9,246.1 123.9,246.7 124.8,247.2 125.7,247.8 126.7,248.3 127.6,248.9 128.5,249.4 129.5,250.0 130.4,250.6 131.3,251.1 132.3,251.7 133.2,252.2 134.1,252.8 135.1,253.3 136.0,253.9 136.9,254.4 137.9,255.0 138.8,255.6 139.7,256.1 140.7,256.7 141.6,257.2 142.5,257.8 143.5,258.3 144.4,258.9 145.3,259.4 146.3,260.0 147.2,260.6 148.1,261.1 149.1,261.7 150.0,262.2 150.9,262.8 151.9,263.3 152.8,263.9 153.7,264.4 154.7,265.0 155.6,265.6 156.5,266.1 157.5,266.7 158.4,267.2 159.3,267.8 160.3,268.3 161.2,268.9 162.1,269.4 163.1,270.0 164.0,270.6 164.9,271.1 165.9,271.7 166.8,272.2 167.7,272.8 168.7,273.3 169.6,273.9 170.5,274.4 171.5,275.0 172.4,275.6 173.3,276.1 174.3,276.7 175.2,277.2 176.1,277.7 177.1,278.2 178.0,278.8 178.9,279.2 179.9,279.7 180.8,280.2 181.7,280.7 182.7,281.1 183.6,281.6 184.5,282.0 185.5,282.4 186.4,282.8 187.3,283.2 188.3,283.6 189.2,283.9 190.1,284.3 191.1,284.7 192.0,285.0 192.9,285.3 193.9,285.6 194.8,285.9 195.7,286.2 196.7,286.5 197.6,286.8 198.5,287.1 199.5,287.3 200.4,287.6 201.3,287.8 202.3,288.0 203.2,288.2 204.1,288.4 205.1,288.6 206.0,288.8 206.9,288.9 207.9,289.1 208.8,289.2 209.7,289.3 210.7,289.4 211.6,289.6 212.5,289.6 213.5,289.7 214.4,289.8 215.3,289.9 216.3,289.9 217.2,289.9 218.1,290.0 219.1,290.0 220.0,290.0 220.9,290.0 221.9,290.0 222.8,289.9 223.7,289.9 224.7,289.9 225.6,289.8 226.5,289.7 227.5,289.6 228.4,289.6 229.3,289.4 230.3,289.3 231.2,289.2 232.1,289.1 233.1,288.9 234.0,288.8 234.9,288.6 235.9,288.4 236.8,288.2 237.7,288.0 238.7,287.8 239.6,287.6 240.5,287.3 241.5,287.1 242.4,286.8 243.3,286.5 244.3,286.2 245.2,285.9 246.1,285.6 247.1,285.3 248.0,285.0 248.9,284.7 249.9,284.3 250.8,283.9 251.7,283.6 252.7,283.2 253.6,282.8 254.5,282.4 255.5,282.0 256.4,281.6 257.3,281.1 258.3,280.7 259.2,280.2 260.1,279.7 261.1,279.2 262.0,278.8 262.9,278.2 263.9,277.7 264.8,277.2 265.7,276.7 266.7,276.1 267.6,275.6 268.5,275.0 269.5,274.4 270.4,273.9 271.3,273.3 272.3,272.8 273.2,272.2 274.1,271.7 275.1,271.1 276.0,270.6 276.9,270.0 277.9,269.4 278.8,268.9 279.7,268.3 280.7,267.8 281.6,267.2 282.5,266.7 283.5,266.1 284.4,265.6 285.3,265.0 286.3,264.4 287.2,263.9 288.1,263.3 289.1,262.8 290.0,262.2 290.9,261.7 291.9,261.1 292.8,260.6 293.7,260.0 294.7,259.4 295.6,258.9 296.5,258.3 297.5,257.8 298.4,257.2 299.3,256.7 300.3,256.1 301.2,255.6 302.1,255.0 303.1,254.4 304.0,253.9 304.9,253.3 305.9,252.8 306.8,252.2 307.7,251.7 308.7,251.1 309.6,250.6 310.5,250.0 311.5,249.4 312.4,248.9 313.3,248.3 314.3,247.8 315.2,247.2 316.1,246.7 317.1,246.1 318.0,245.6 318.9,245.0 319.9,244.4 320.8,243.9 321.7,243.3 322.7,242.8 323.6,242.2 324.5,241.7 325.5,241.1 326.4,240.6 327.3,240.0 328.3,239.4 329.2,238.9 330.1,238.3 331.1,237.8 332.0,237.2 332.9,236.7 333.9,236.1 334.8,235.6 335.7,235.0 336.7,234.4 337.6,233.9 338.5,233.3 339.5,232.8 340.4,232.2 341.3,231.7 342.3,231.1 343.2,230.6 344.1,230.0 345.1,229.4 346.0,228.9 346.9,228.3 347.9,227.8 348.8,227.2 349.7,226.7 350.7,226.1 351.6,225.6 352.5,225.0 353.5,224.4 354.4,223.9 355.3,223.3 356.3,222.8 357.2,222.2 358.1,221.7 359.1,221.1 360.0,220.6" stroke-dasharray="7 4"/><polyline class="dg-line-accent" points="80.0,40.0 80.9,43.3 81.9,46.6 82.8,49.9 83.7,53.2 84.7,56.4 85.6,59.6 86.5,62.8 87.5,66.0 88.4,69.1 89.3,72.2 90.3,75.3 91.2,78.4 92.1,81.5 93.1,84.5 94.0,87.5 94.9,90.5 95.9,93.5 96.8,96.4 97.7,99.3 98.7,102.2 99.6,105.1 100.5,108.0 101.5,110.8 102.4,113.6 103.3,116.4 104.3,119.2 105.2,121.9 106.1,124.6 107.1,127.3 108.0,130.0 108.9,132.7 109.9,135.3 110.8,137.9 111.7,140.5 112.7,143.1 113.6,145.6 114.5,148.1 115.5,150.6 116.4,153.1 117.3,155.6 118.3,158.0 119.2,160.4 120.1,162.8 121.1,165.2 122.0,167.5 122.9,169.8 123.9,172.1 124.8,174.4 125.7,176.7 126.7,178.9 127.6,181.1 128.5,183.3 129.5,185.5 130.4,187.6 131.3,189.7 132.3,191.8 133.2,193.9 134.1,196.0 135.1,198.0 136.0,200.0 136.9,202.0 137.9,204.0 138.8,205.9 139.7,207.8 140.7,209.7 141.6,211.6 142.5,213.5 143.5,215.3 144.4,217.1 145.3,218.9 146.3,220.7 147.2,222.4 148.1,224.1 149.1,225.8 150.0,227.5 150.9,229.2 151.9,230.8 152.8,232.4 153.7,234.0 154.7,235.6 155.6,237.1 156.5,238.6 157.5,240.1 158.4,241.6 159.3,243.1 160.3,244.5 161.2,245.9 162.1,247.3 163.1,248.7 164.0,250.0 164.9,251.3 165.9,252.6 166.8,253.9 167.7,255.2 168.7,256.4 169.6,257.6 170.5,258.8 171.5,260.0 172.4,261.1 173.3,262.2 174.3,263.3 175.2,264.4 176.1,265.5 177.1,266.5 178.0,267.5 178.9,268.5 179.9,269.5 180.8,270.4 181.7,271.3 182.7,272.2 183.6,273.1 184.5,274.0 185.5,274.8 186.4,275.6 187.3,276.4 188.3,277.2 189.2,277.9 190.1,278.6 191.1,279.3 192.0,280.0 192.9,280.7 193.9,281.3 194.8,281.9 195.7,282.5 196.7,283.1 197.6,283.6 198.5,284.1 199.5,284.6 200.4,285.1 201.3,285.6 202.3,286.0 203.2,286.4 204.1,286.8 205.1,287.2 206.0,287.5 206.9,287.8 207.9,288.1 208.8,288.4 209.7,288.7 210.7,288.9 211.6,289.1 212.5,289.3 213.5,289.5 214.4,289.6 215.3,289.7 216.3,289.8 217.2,289.9 218.1,290.0 219.1,290.0 220.0,290.0 220.9,290.0 221.9,290.0 222.8,289.9 223.7,289.8 224.7,289.7 225.6,289.6 226.5,289.5 227.5,289.3 228.4,289.1 229.3,288.9 230.3,288.7 231.2,288.4 232.1,288.1 233.1,287.8 234.0,287.5 234.9,287.2 235.9,286.8 236.8,286.4 237.7,286.0 238.7,285.6 239.6,285.1 240.5,284.6 241.5,284.1 242.4,283.6 243.3,283.1 244.3,282.5 245.2,281.9 246.1,281.3 247.1,280.7 248.0,280.0 248.9,279.3 249.9,278.6 250.8,277.9 251.7,277.2 252.7,276.4 253.6,275.6 254.5,274.8 255.5,274.0 256.4,273.1 257.3,272.2 258.3,271.3 259.2,270.4 260.1,269.5 261.1,268.5 262.0,267.5 262.9,266.5 263.9,265.5 264.8,264.4 265.7,263.3 266.7,262.2 267.6,261.1 268.5,260.0 269.5,258.8 270.4,257.6 271.3,256.4 272.3,255.2 273.2,253.9 274.1,252.6 275.1,251.3 276.0,250.0 276.9,248.7 277.9,247.3 278.8,245.9 279.7,244.5 280.7,243.1 281.6,241.6 282.5,240.1 283.5,238.6 284.4,237.1 285.3,235.6 286.3,234.0 287.2,232.4 288.1,230.8 289.1,229.2 290.0,227.5 290.9,225.8 291.9,224.1 292.8,222.4 293.7,220.7 294.7,218.9 295.6,217.1 296.5,215.3 297.5,213.5 298.4,211.6 299.3,209.7 300.3,207.8 301.2,205.9 302.1,204.0 303.1,202.0 304.0,200.0 304.9,198.0 305.9,196.0 306.8,193.9 307.7,191.8 308.7,189.7 309.6,187.6 310.5,185.5 311.5,183.3 312.4,181.1 313.3,178.9 314.3,176.7 315.2,174.4 316.1,172.1 317.1,169.8 318.0,167.5 318.9,165.2 319.9,162.8 320.8,160.4 321.7,158.0 322.7,155.6 323.6,153.1 324.5,150.6 325.5,148.1 326.4,145.6 327.3,143.1 328.3,140.5 329.2,137.9 330.1,135.3 331.1,132.7 332.0,130.0 332.9,127.3 333.9,124.6 334.8,121.9 335.7,119.2 336.7,116.4 337.6,113.6 338.5,110.8 339.5,108.0 340.4,105.1 341.3,102.2 342.3,99.3 343.2,96.4 344.1,93.5 345.1,90.5 346.0,87.5 346.9,84.5 347.9,81.5 348.8,78.4 349.7,75.3 350.7,72.2 351.6,69.1 352.5,66.0 353.5,62.8 354.4,59.6 355.3,56.4 356.3,53.2 357.2,49.9 358.1,46.6 359.1,43.3 360.0,40.0"/><polyline class="dg-line" points="470.0,185.8 605.0,185.8 605.0,144.2 740.0,144.2"/><polyline class="dg-line" points="470.0,185.8 470.9,185.8 471.8,185.8 472.7,185.8 473.6,185.8 474.5,185.8 475.4,185.8 476.3,185.8 477.2,185.8 478.1,185.8 479.0,185.8 479.9,185.8 480.8,185.8 481.7,185.8 482.6,185.8 483.5,185.8 484.4,185.8 485.3,185.8 486.2,185.8 487.1,185.8 488.0,185.8 488.9,185.8 489.8,185.8 490.7,185.8 491.6,185.8 492.5,185.8 493.4,185.8 494.3,185.8 495.2,185.8 496.1,185.8 497.0,185.8 497.9,185.8 498.8,185.8 499.7,185.8 500.6,185.8 501.5,185.8 502.4,185.8 503.3,185.8 504.2,185.8 505.1,185.8 506.0,185.8 506.9,185.8 507.8,185.8 508.7,185.8 509.6,185.8 510.5,185.8 511.4,185.8 512.3,185.8 513.2,185.8 514.1,185.8 515.0,185.8 515.9,185.8 516.8,185.8 517.7,185.8 518.6,185.8 519.5,185.8 520.4,185.8 521.3,185.8 522.2,185.8 523.1,185.8 524.0,185.8 524.9,185.8 525.8,185.8 526.7,185.8 527.6,185.8 528.5,185.8 529.4,185.8 530.3,185.8 531.2,185.8 532.1,185.8 533.0,185.8 533.9,185.8 534.8,185.8 535.7,185.8 536.6,185.8 537.5,185.8 538.4,185.8 539.3,185.8 540.2,185.8 541.1,185.8 542.0,185.8 542.9,185.8 543.8,185.8 544.7,185.8 545.6,185.8 546.5,185.8 547.4,185.8 548.3,185.8 549.2,185.8 550.1,185.8 551.0,185.8 551.9,185.8 552.8,185.8 553.7,185.8 554.6,185.8 555.5,185.8 556.4,185.8 557.3,185.8 558.2,185.8 559.1,185.8 560.0,185.8 560.9,185.4 561.8,185.0 562.7,184.6 563.6,184.2 564.5,183.8 565.4,183.3 566.3,182.9 567.2,182.5 568.1,182.1 569.0,181.7 569.9,181.2 570.8,180.8 571.7,180.4 572.6,180.0 573.5,179.6 574.4,179.2 575.3,178.8 576.2,178.3 577.1,177.9 578.0,177.5 578.9,177.1 579.8,176.7 580.7,176.2 581.6,175.8 582.5,175.4 583.4,175.0 584.3,174.6 585.2,174.2 586.1,173.8 587.0,173.3 587.9,172.9 588.8,172.5 589.7,172.1 590.6,171.7 591.5,171.2 592.4,170.8 593.3,170.4 594.2,170.0 595.1,169.6 596.0,169.2 596.9,168.8 597.8,168.3 598.7,167.9 599.6,167.5 600.5,167.1 601.4,166.7 602.3,166.2 603.2,165.8 604.1,165.4 605.0,165.0 605.9,164.6 606.8,164.2 607.7,163.8 608.6,163.3 609.5,162.9 610.4,162.5 611.3,162.1 612.2,161.7 613.1,161.2 614.0,160.8 614.9,160.4 615.8,160.0 616.7,159.6 617.6,159.2 618.5,158.8 619.4,158.3 620.3,157.9 621.2,157.5 622.1,157.1 623.0,156.7 623.9,156.2 624.8,155.8 625.7,155.4 626.6,155.0 627.5,154.6 628.4,154.2 629.3,153.8 630.2,153.3 631.1,152.9 632.0,152.5 632.9,152.1 633.8,151.7 634.7,151.2 635.6,150.8 636.5,150.4 637.4,150.0 638.3,149.6 639.2,149.2 640.1,148.7 641.0,148.3 641.9,147.9 642.8,147.5 643.7,147.1 644.6,146.7 645.5,146.2 646.4,145.8 647.3,145.4 648.2,145.0 649.1,144.6 650.0,144.2 650.9,144.2 651.8,144.2 652.7,144.2 653.6,144.2 654.5,144.2 655.4,144.2 656.3,144.2 657.2,144.2 658.1,144.2 659.0,144.2 659.9,144.2 660.8,144.2 661.7,144.2 662.6,144.2 663.5,144.2 664.4,144.2 665.3,144.2 666.2,144.2 667.1,144.2 668.0,144.2 668.9,144.2 669.8,144.2 670.7,144.2 671.6,144.2 672.5,144.2 673.4,144.2 674.3,144.2 675.2,144.2 676.1,144.2 677.0,144.2 677.9,144.2 678.8,144.2 679.7,144.2 680.6,144.2 681.5,144.2 682.4,144.2 683.3,144.2 684.2,144.2 685.1,144.2 686.0,144.2 686.9,144.2 687.8,144.2 688.7,144.2 689.6,144.2 690.5,144.2 691.4,144.2 692.3,144.2 693.2,144.2 694.1,144.2 695.0,144.2 695.9,144.2 696.8,144.2 697.7,144.2 698.6,144.2 699.5,144.2 700.4,144.2 701.3,144.2 702.2,144.2 703.1,144.2 704.0,144.2 704.9,144.2 705.8,144.2 706.7,144.2 707.6,144.2 708.5,144.2 709.4,144.2 710.3,144.2 711.2,144.2 712.1,144.2 713.0,144.2 713.9,144.2 714.8,144.2 715.7,144.2 716.6,144.2 717.5,144.2 718.4,144.2 719.3,144.2 720.2,144.2 721.1,144.2 722.0,144.2 722.9,144.2 723.8,144.2 724.7,144.2 725.6,144.2 726.5,144.2 727.4,144.2 728.3,144.2 729.2,144.2 730.1,144.2 731.0,144.2 731.9,144.2 732.8,144.2 733.7,144.2 734.6,144.2 735.5,144.2 736.4,144.2 737.3,144.2 738.2,144.2 739.1,144.2 740.0,144.2" stroke-dasharray="7 4"/><polyline class="dg-line-accent" points="470.0,290.0 470.9,289.2 471.8,288.3 472.7,287.5 473.6,286.7 474.5,285.8 475.4,285.0 476.3,284.2 477.2,283.3 478.1,282.5 479.0,281.7 479.9,280.8 480.8,280.0 481.7,279.2 482.6,278.3 483.5,277.5 484.4,276.7 485.3,275.8 486.2,275.0 487.1,274.2 488.0,273.3 488.9,272.5 489.8,271.7 490.7,270.8 491.6,270.0 492.5,269.2 493.4,268.3 494.3,267.5 495.2,266.7 496.1,265.8 497.0,265.0 497.9,264.2 498.8,263.3 499.7,262.5 500.6,261.7 501.5,260.8 502.4,260.0 503.3,259.2 504.2,258.3 505.1,257.5 506.0,256.7 506.9,255.8 507.8,255.0 508.7,254.2 509.6,253.3 510.5,252.5 511.4,251.7 512.3,250.8 513.2,250.0 514.1,249.2 515.0,248.3 515.9,247.5 516.8,246.7 517.7,245.8 518.6,245.0 519.5,244.2 520.4,243.3 521.3,242.5 522.2,241.7 523.1,240.8 524.0,240.0 524.9,239.2 525.8,238.3 526.7,237.5 527.6,236.7 528.5,235.8 529.4,235.0 530.3,234.2 531.2,233.3 532.1,232.5 533.0,231.7 533.9,230.8 534.8,230.0 535.7,229.2 536.6,228.3 537.5,227.5 538.4,226.7 539.3,225.8 540.2,225.0 541.1,224.2 542.0,223.3 542.9,222.5 543.8,221.7 544.7,220.8 545.6,220.0 546.5,219.2 547.4,218.3 548.3,217.5 549.2,216.7 550.1,215.8 551.0,215.0 551.9,214.2 552.8,213.3 553.7,212.5 554.6,211.7 555.5,210.8 556.4,210.0 557.3,209.2 558.2,208.3 559.1,207.5 560.0,206.7 560.9,205.8 561.8,205.0 562.7,204.2 563.6,203.3 564.5,202.5 565.4,201.7 566.3,200.8 567.2,200.0 568.1,199.2 569.0,198.3 569.9,197.5 570.8,196.7 571.7,195.8 572.6,195.0 573.5,194.2 574.4,193.3 575.3,192.5 576.2,191.7 577.1,190.8 578.0,190.0 578.9,189.2 579.8,188.3 580.7,187.5 581.6,186.7 582.5,185.8 583.4,185.0 584.3,184.2 585.2,183.3 586.1,182.5 587.0,181.7 587.9,180.8 588.8,180.0 589.7,179.2 590.6,178.3 591.5,177.5 592.4,176.7 593.3,175.8 594.2,175.0 595.1,174.2 596.0,173.3 596.9,172.5 597.8,171.7 598.7,170.8 599.6,170.0 600.5,169.2 601.4,168.3 602.3,167.5 603.2,166.7 604.1,165.8 605.0,165.0 605.9,164.2 606.8,163.3 607.7,162.5 608.6,161.7 609.5,160.8 610.4,160.0 611.3,159.2 612.2,158.3 613.1,157.5 614.0,156.7 614.9,155.8 615.8,155.0 616.7,154.2 617.6,153.3 618.5,152.5 619.4,151.7 620.3,150.8 621.2,150.0 622.1,149.2 623.0,148.3 623.9,147.5 624.8,146.7 625.7,145.8 626.6,145.0 627.5,144.2 628.4,143.3 629.3,142.5 630.2,141.7 631.1,140.8 632.0,140.0 632.9,139.2 633.8,138.3 634.7,137.5 635.6,136.7 636.5,135.8 637.4,135.0 638.3,134.2 639.2,133.3 640.1,132.5 641.0,131.7 641.9,130.8 642.8,130.0 643.7,129.2 644.6,128.3 645.5,127.5 646.4,126.7 647.3,125.8 648.2,125.0 649.1,124.2 650.0,123.3 650.9,122.5 651.8,121.7 652.7,120.8 653.6,120.0 654.5,119.2 655.4,118.3 656.3,117.5 657.2,116.7 658.1,115.8 659.0,115.0 659.9,114.2 660.8,113.3 661.7,112.5 662.6,111.7 663.5,110.8 664.4,110.0 665.3,109.2 666.2,108.3 667.1,107.5 668.0,106.7 668.9,105.8 669.8,105.0 670.7,104.2 671.6,103.3 672.5,102.5 673.4,101.7 674.3,100.8 675.2,100.0 676.1,99.2 677.0,98.3 677.9,97.5 678.8,96.7 679.7,95.8 680.6,95.0 681.5,94.2 682.4,93.3 683.3,92.5 684.2,91.7 685.1,90.8 686.0,90.0 686.9,89.2 687.8,88.3 688.7,87.5 689.6,86.7 690.5,85.8 691.4,85.0 692.3,84.2 693.2,83.3 694.1,82.5 695.0,81.7 695.9,80.8 696.8,80.0 697.7,79.2 698.6,78.3 699.5,77.5 700.4,76.7 701.3,75.8 702.2,75.0 703.1,74.2 704.0,73.3 704.9,72.5 705.8,71.7 706.7,70.8 707.6,70.0 708.5,69.2 709.4,68.3 710.3,67.5 711.2,66.7 712.1,65.8 713.0,65.0 713.9,64.2 714.8,63.3 715.7,62.5 716.6,61.7 717.5,60.8 718.4,60.0 719.3,59.2 720.2,58.3 721.1,57.5 722.0,56.7 722.9,55.8 723.8,55.0 724.7,54.2 725.6,53.3 726.5,52.5 727.4,51.7 728.3,50.8 729.2,50.0 730.1,49.2 731.0,48.3 731.9,47.5 732.8,46.7 733.7,45.8 734.6,45.0 735.5,44.2 736.4,43.3 737.3,42.5 738.2,41.7 739.1,40.8 740.0,40.0"/><line class="dg-line-accent" x1="150.0" y1="358.0" x2="178.0" y2="358.0"/><text class="dg-text" x="186" y="363">MSE: r²</text><line class="dg-line" x1="320.0" y1="358.0" x2="348.0" y2="358.0"/><text class="dg-text" x="356" y="363">MAE: |r|</text><line class="dg-line" x1="490.0" y1="358.0" x2="518.0" y2="358.0" stroke-dasharray="7 4"/><text class="dg-text" x="526" y="363">Huber, δ = 1</text></svg>
</div>
<figcaption>MSE, MAE and Huber loss (left) and their gradients (right), computed from the formulas for residuals from −3 to 3. The shaded band is |r| ≤ δ, where Huber behaves like MSE.</figcaption>
</figure>

| Loss          | Formula per example                  | Best constant           | Outliers                       | Gradient                        | Use it when                                                     |
| ------------- | ------------------------------------ | ----------------------- | ------------------------------ | ------------------------------- | --------------------------------------------------------------- |
| MSE           | r²                                   | Mean                    | Very sensitive                 | 2r: large for large errors      | Errors are roughly Gaussian; large errors really are much worse |
| MAE           | \|r\|                                | Median                  | Robust                         | ±1 (undefined at 0)             | Heavy-tailed noise, outliers you do not trust                   |
| Huber         | ½r² if \|r\| ≤ δ, else δ(\|r\| − ½δ) | Between median and mean | Robust                         | r, capped at ±δ                 | You want MSE's smoothness without its outlier sensitivity       |
| Cross-entropy | −log p(true class)                   | Class frequencies       | Punishes confident errors hard | p − y (with softmax or sigmoid) | Classification; you need probabilities                          |

**Other losses worth recognizing:**

- **Hinge loss**, max(0, 1 − y·f(x)), used by SVMs (Chapter 2): zero once an example is beyond the margin.
- **Quantile (pinball) loss** predicts a chosen quantile rather than the mean; quantile 0.5 is MAE. Used for forecast intervals ("90% of deliveries arrive within this time").
- **Log-cosh loss**, log(cosh(r)): a smooth Huber-like loss, twice differentiable everywhere.
- **Focal loss** and **class-balanced loss** for imbalanced classification (Chapter 6).
- **KL divergence** for matching distributions, as in knowledge distillation and variational autoencoders (8.2).
- **Contrastive and triplet losses** for embeddings: pull matching pairs together and push others apart, as in CLIP and retrieval models.
- **Ranking losses** (pairwise, listwise) for search and recommendations, where the order matters more than each score.

## 8.2 Core math

**Losses come from likelihoods.** If you assume y = f(x) + ε and maximize the likelihood of the data, the loss you minimize is the negative log-likelihood of the noise model:

| Noise assumed            | Negative log-likelihood per example | Loss                 |
| ------------------------ | ----------------------------------- | -------------------- |
| Gaussian, fixed variance | (y − ŷ)² / 2σ² + constant           | MSE                  |
| Laplace                  | \|y − ŷ\| / b + constant            | MAE                  |
| Bernoulli (0/1 label)    | −\[y log p + (1 − y) log(1 − p)\]   | Binary cross-entropy |
| Categorical (K classes)  | −log p_y                            | Cross-entropy        |

So choosing MSE is choosing to believe the noise is Gaussian; heavy-tailed noise calls for MAE or Huber.

**Best constant predictions.** Minimizing Σ(y_i − c)² gives c = mean(y): set the derivative −2Σ(y_i − c) to zero. Minimizing Σ|y_i − c| gives c = median(y): the derivative is (number of points below c) minus (number above), which is zero at the median. For y = \[1, 2, 3, 4, 100\], MSE picks 22 and MAE picks 3. A model trained with MSE predicts the conditional mean of y given x; one trained with MAE predicts the conditional median.

**Cross-entropy, entropy and KL divergence.** For a true distribution p and a predicted distribution q over classes:

$$
H(p) = -\sum_k p_k \log p_k, \qquad H(p, q) = -\sum_k p_k \log q_k, \qquad D_{\text{KL}}(p \,\|\, q) = \sum_k p_k \log\frac{p_k}{q_k} = H(p, q) - H(p)
$$

H(p) is fixed by the data, so minimizing cross-entropy is the same as minimizing the KL divergence from the true distribution to the model's. With a one-hot label, H(p) = 0 and cross-entropy reduces to −log q_y. KL divergence is never negative and is zero only when q = p, but it is **not symmetric**: D_KL(p‖q) ≠ D_KL(q‖p), so it is not a true distance. Example: p = (0.7, 0.2, 0.1), q = (0.5, 0.3, 0.2) gives H(p) = 0.802, H(p, q) = 0.887 and KL = 0.085 nats.

**Softmax and its gradient.** With logits z and p = softmax(z), the cross-entropy gradient with respect to the logits is strikingly simple, with y the one-hot label:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad L = -\log p_y, \qquad \frac{\partial L}{\partial z_k} = p_k - y_k
$$

The same "prediction minus target" form appears for sigmoid with binary cross-entropy and for linear regression with MSE, which is no coincidence: all three are generalized linear models with their natural (canonical) link.

**Why not MSE for classification?** With a sigmoid output, MSE's gradient is (p − y)·p(1 − p). When the model is confidently wrong (p near 0 for a true 1), the factor p(1 − p) is near 0, so learning stalls exactly when it should be fastest. Cross-entropy's gradient, p − y, stays large.

**Numerical stability: the log-sum-exp trick.** Computing softmax naively overflows for large logits: e^1000 is infinite in floating point. Subtract the maximum logit first, which changes nothing mathematically:

$$
\log p_k = z_k - \log\sum_j e^{z_j} = (z_k - m) - \log\sum_j e^{z_j - m}, \qquad m = \max_j z_j
$$

This is why frameworks take raw logits in losses such as PyTorch's `CrossEntropyLoss` and `BCEWithLogitsLoss` instead of probabilities.

**Label smoothing.** Replace the one-hot target with a softened one, giving ε/K to every class. With K = 3 and ε = 0.1, the target (1, 0, 0) becomes (0.933, 0.033, 0.033). The model can no longer drive the true-class logit to infinity, which improves calibration and acts as a regularizer:

$$
y_k^{\text{LS}} = (1 - \varepsilon)\, y_k + \frac{\varepsilon}{K}
$$

**Huber loss and its gradient.** With threshold δ:

$$
L_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \le \delta \\ \delta\big(|r| - \tfrac{1}{2}\delta\big) & |r| > \delta \end{cases}, \qquad \frac{\partial L_\delta}{\partial r} = \operatorname{clip}(r, -\delta, \delta)
$$

The two pieces meet with matching value and slope at |r| = δ, so the loss is smooth. The gradient is the residual clipped to ±δ: an outlier can pull on the fit only with force δ, however wrong it is. The same idea, under the name **smooth L1 loss**, is standard for bounding-box regression in object detection.

**Quantile loss** for quantile τ between 0 and 1 penalizes under-prediction with weight τ and over-prediction with weight 1 − τ, so the minimizer is the τ-th quantile:

$$
L_\tau(r) = \max\big(\tau\, r,\ (\tau - 1)\, r\big), \qquad r = y - \hat y
$$

## 8.3 Code

**Stable cross-entropy from logits, and why the naive version fails:**

```python
import numpy as np

def cross_entropy(logits, labels):
    z = logits - logits.max(axis=1, keepdims=True)              # log-sum-exp trick
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(labels)), labels].mean()

logits = np.array([[2.0, 1.0, 0.1], [1000.0, 0.0, -1000.0]])
labels = np.array([0, 0])
print(cross_entropy(logits, labels))                            # 0.2085

naive = -np.log(np.exp(logits[1, 0]) / np.exp(logits[1]).sum())
print(naive)                                                    # nan: exp(1000) overflowed
```

**MSE, MAE and Huber on data with outliers.** 200 points from y = 3x + 5 + noise; then 20 of them (10%) get 30 to 60 added, like corrupted sensor readings.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(200, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 1, 200)
idx = rng.choice(200, 20, replace=False)
y_out = y.copy()
y_out[idx] += rng.uniform(30, 60, 20)                           # 10% gross outliers

models = {
    "MSE (ordinary least squares)": LinearRegression(),
    "Huber": HuberRegressor(epsilon=1.35),
    "MAE (median regression)": QuantileRegressor(quantile=0.5, alpha=0, solver="highs"),
}
for name, m in models.items():
    m.fit(X, y_out)
    print(name, m.coef_[0], m.intercept_)
```

| Loss                         | Slope (true 3) | Intercept (true 5) |
| ---------------------------- | -------------- | ------------------ |
| MSE (ordinary least squares) | 2.115          | 14.335             |
| Huber                        | 2.941          | 5.354              |
| MAE (median regression)      | 2.933          | 5.307              |

Ten percent bad data pushed the least-squares intercept from 5 to over 14; Huber and MAE barely moved. scikit-learn's `HuberRegressor` uses `epsilon` as its threshold on scaled residuals; 1.35 is the classic choice that keeps 95% of least-squares efficiency when the noise really is Gaussian.

**Quantile regression for prediction intervals.** `QuantileRegressor(quantile=0.9)` fitted on the clean data gives a line under which 90.5% of the training points fall, close to the 90% target. Fitting quantiles 0.05 and 0.95 gives a 90% prediction interval.

## 8.4 In system design

- **Match the loss to the business quantity.** Delivery-time estimates shown to customers are judged on late deliveries, so predict a high quantile (say the 90th) with quantile loss rather than the mean. Demand forecasts that drive inventory often use quantile loss too, because understocking and overstocking cost different amounts.
- **The loss decides what a prediction means.** An MSE model predicts the average, so on skewed targets (revenue, time spent) its typical prediction is pulled up by rare large values. Training on log(y) with MSE predicts the geometric mean instead; say which one the product needs.
- **Probabilities need log loss.** Ad bidding, risk pricing and blending scores from several models need calibrated probabilities, so train with cross-entropy and monitor calibration in production, not just ranking metrics.
- **Train on a surrogate, evaluate on the metric.** You cannot backpropagate through accuracy, NDCG or revenue. Train on cross-entropy or a ranking loss, then pick models and thresholds by the real metric on validation data.
- **Multi-task models** add several losses (click, like, share) with weights. The weights are product decisions as much as technical ones, and they must be revisited when one task's loss scale changes.
- **Distillation** trains a small student model on a large teacher's soft probabilities with a KL-divergence loss, often with a temperature that softens both distributions, plus the usual cross-entropy on true labels. It is a standard way to make models cheap enough to serve.
- **Robust losses as a data-quality guard.** If labels come from noisy sources (crowd workers, logs), Huber, MAE or label smoothing limit how much bad labels can drag the model.

## 8.5 Interview Q&A

**Q1. Why is cross-entropy used for classification instead of MSE?** Cross-entropy is the negative log-likelihood of a categorical model, so minimizing it is maximum likelihood. With a softmax or sigmoid, its gradient is p − y, which stays large when the model is confidently wrong; MSE through a sigmoid has a gradient scaled by p(1 − p), which vanishes in exactly that case, and the MSE loss is non-convex in the logits.

**Q2. MSE or MAE: which would you use, and what does each predict?** MSE predicts the conditional mean and suits roughly Gaussian errors where large mistakes are much worse. MAE predicts the conditional median and is robust to outliers and heavy tails. Choose by the noise and by which summary the business needs.

**Q3. What is Huber loss and how do you choose δ?** It is quadratic for residuals up to δ and linear beyond, so it is smooth near the optimum but limits each outlier's pull to δ. Choose δ near the scale of typical, trustworthy errors, for example a multiple of a robust noise estimate such as the median absolute deviation, or tune it on validation data.

**Q4. What is the relationship between cross-entropy and KL divergence?** H(p, q) = H(p) + D_KL(p‖q). The entropy of the true labels does not depend on the model, so minimizing cross-entropy minimizes the KL divergence. KL is non-negative, zero only when the distributions match, and not symmetric.

**Q5. Derive the gradient of softmax cross-entropy with respect to the logits.** L = −z_y + log Σ_j e^{z_j}. Differentiating, ∂L/∂z_k = −\[k = y\] + e^{z_k}/Σ_j e^{z_j} = p_k − y_k. So the gradient is the predicted distribution minus the one-hot label.

**Q6. Why do PyTorch losses take logits rather than probabilities?** Computing log(softmax(z)) in two steps can overflow in exp or produce log(0). Fusing them with the log-sum-exp trick (subtract the maximum logit) is numerically stable, so `CrossEntropyLoss` and `BCEWithLogitsLoss` take raw logits.

**Q7. What does label smoothing do, and when can it hurt?** It replaces one-hot targets with (1 − ε) on the true class and ε/K spread over all classes. It stops logits from growing without bound, improves calibration and generalization, and is standard in image and translation models. It can hurt knowledge distillation (teachers trained with it pass on less information about class similarity) and tasks that need very confident probabilities.

**Q8. Your regression model's RMSE is good but users complain about large errors on a few items. What do you look at?** The error distribution, not just its average: tail percentiles of the absolute error, and the error by segment. If certain items are systematically mis-predicted, look for missing features or bad labels; if the business cares about the tail, train for a high quantile or weight the loss toward those items.

**Q9. How would you train a model to output a 90% prediction interval?** Train two quantile models (or one model with two outputs) with quantile loss at τ = 0.05 and τ = 0.95, then check on validation data that about 90% of true values fall inside. Conformal prediction can adjust the interval to guarantee that coverage.

**Q10. What loss would you use for a ranking problem such as search results?** Pointwise losses (cross-entropy on click or no click) are simple and work well with large data. Pairwise losses (RankNet, BPR) learn which of two items should rank higher. Listwise losses (LambdaRank, softmax over the list) optimize the whole ordering and track ranking metrics like NDCG more closely.

## 8.6 Cheat sheet and pitfalls

**Remember:**

- MSE ↔ Gaussian noise ↔ conditional mean; MAE ↔ Laplace noise ↔ conditional median; quantile loss ↔ any quantile.
- Cross-entropy = negative log-likelihood; H(p, q) = H(p) + KL(p‖q).
- Softmax + cross-entropy gradient = p − y; sigmoid + binary cross-entropy gradient = p − y.
- Huber: ½r² inside δ, linear outside; gradient = r clipped to ±δ.
- Stability: subtract the maximum logit; pass logits to fused losses.
- Label smoothing: target (1 − ε)y + ε/K.
- Train on a differentiable surrogate; select on the business metric.

**Pitfalls that cost candidates:**

- Using MSE for classification, or saying cross-entropy and log loss are different things.
- Forgetting that MSE predicts the mean, then being surprised by poor typical-case predictions on skewed data.
- Applying softmax before a loss that expects logits, which double-applies it.
- Calling KL divergence a distance or treating it as symmetric.
- Comparing RMSE and MAE values directly as if they were on the same footing.
- Choosing the loss without asking what decision the prediction feeds.

## 8.7 Further reading

<ol start="15">
  <li><a href="https://en.wikipedia.org/wiki/Cross_entropy">Cross-entropy</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Mean_squared_error">Mean squared error</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Mean_absolute_error">Mean absolute error</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Huber_loss">Huber loss</a> (Wikipedia)</li>
</ol>
