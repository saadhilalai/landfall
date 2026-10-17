## Training runs

| Level | Seed | Started from | Validation at start | Best validation | Mastered? | Practice steps | Minutes |
|---|---|---|---|---|---|---|---|
| 1 Limited fuel tank | 42 | Final AI | 83.5% | 91.5% | yes | 1,500,000 | 6.3 |
| 1 Limited fuel tank | 142 | Final AI | 93.0% | 93.0% | yes | 16 | 0.1 |
| 1 Limited fuel tank | 242 | Final AI | 78.0% | 92.0% | yes | 1,000,000 | 2.7 |
| 2 Landing pad in a random spot | 42 | Level 1 brain | 88.0% | 91.0% | yes | 5,000,000 | 15.4 |
| 2 Landing pad in a random spot | 142 | Level 1 brain | 86.0% | 91.0% | yes | 2,000,000 | 6.1 |
| 2 Landing pad in a random spot | 242 | Level 1 brain | 79.5% | 91.0% | yes | 2,500,000 | 6.4 |
| 3a Obstacles, walls up to 2 m | 42 | Level 2 brain | 88.0% | 90.0% | yes | 1,500,000 | 5.0 |
| 3a Obstacles, walls up to 2 m | 142 | Level 2 brain | 87.5% | 90.5% | yes | 3,500,000 | 11.1 |
| 3a Obstacles, walls up to 2 m | 242 | Level 2 brain | 88.0% | 91.0% | yes | 3,000,000 | 8.8 |
| 3b Obstacles, walls up to 3.5 m | 42 | Level 3a brain | 86.5% | 89.0% | no | 12,001,280 | 36.6 |
| 3b Obstacles, walls up to 3.5 m | 142 | Level 3a brain | 86.5% | 90.0% | yes | 5,500,000 | 17.2 |
| 3b Obstacles, walls up to 3.5 m | 242 | Level 3a brain | 88.5% | 90.0% | yes | 4,000,000 | 11.8 |
| 3 Obstacles: rocks and walls | 142 | Level 3b brain | 87.5% | 90.5% | yes | 8,500,000 | 25.2 |
| 3 Obstacles: rocks and walls | 242 | Level 3b brain | 85.5% | 90.5% | yes | 10,500,000 | 31.8 |
| 4a Mission without obstacles | 142 | Level 4a brain | 20.5% | 75.5% | no | 12,001,280 | 33.3 |
| 4a Mission without obstacles | 242 | Level 3 brain | 14.5% | 68.5% | no | 12,001,280 | 33.3 |
| 4a overtime Mission without obstacles | 142 | Level 4a brain | 75.5% | 84.0% | no | 12,001,280 | 32.9 |
| 4a overtime Mission without obstacles | 242 | Level 4a brain | 68.5% | 78.5% | no | 12,001,280 | 33.0 |
| 4a overtime2 Mission without obstacles | 142 | Level 4a_overtime brain | 84.0% | 84.0% | no | 12,001,280 | 31.7 |
| 4a overtime2 Mission without obstacles | 242 | Level 4a_overtime brain | 78.5% | 78.5% | no | 12,001,280 | 31.6 |

## Hidden test (300 landings per row, seeds 40000-40299, run once)

| Brain | Seed | Tested on level | Landed safely (95% CI) | Score | Fuel per landing | Ran dry | Outcomes |
|---|---|---|---|---|---|---|---|
| Final AI (start) | 42 | 1 | 79.0% (74%–83%) | 221.9 | 97 | 28.7% | {'landed': 237, 'crashed': 60, 'landed_off_pad': 2, 'flew_away': 1} |
| Final AI (start) | 42 | 2 | 76.7% (72%–81%) | 216.7 | 98 | 28.3% | {'landed': 230, 'crashed': 66, 'landed_off_pad': 3, 'out_of_time': 1} |
| Final AI (start) | 42 | 3 | 69.0% (64%–74%) | 184.8 | 98 | 25.0% | {'landed': 207, 'hit_obstacle': 41, 'crashed': 51, 'landed_off_pad': 1} |
| Final AI (start) | 42 | 4 | 0.0% (0%–1%) | 105.2 | - | 0.0% | {'hit_obstacle': 192, 'out_of_time': 3, 'stuck_on_first_pad': 95, 'crashed': 10} |
| Final AI (start) | 42 | 4a | 1.3% (1%–3%) | 167.0 | 149 | 0.0% | {'stuck_on_first_pad': 276, 'landed_off_pad': 1, 'crashed': 18, 'landed': 4, 'out_of_time': 1} |
| Level 1 brain | 42 | 1 | 88.3% (84%–91%) | 252.4 | 89 | 11.7% | {'landed': 265, 'crashed': 32, 'landed_off_pad': 3} |
| Level 2 brain | 42 | 1 | 93.0% (90%–95%) | 263.3 | 88 | 8.0% | {'landed': 279, 'crashed': 20, 'flew_away': 1} |
| Level 2 brain | 42 | 2 | 87.0% (83%–90%) | 250.3 | 89 | 12.7% | {'landed': 261, 'crashed': 36, 'landed_off_pad': 2, 'flew_away': 1} |
| Final AI (start) | 142 | 1 | 90.3% (86%–93%) | 254.8 | 93 | 12.3% | {'landed': 271, 'crashed': 24, 'landed_off_pad': 5} |
| Final AI (start) | 142 | 2 | 86.7% (82%–90%) | 239.5 | 94 | 13.3% | {'landed': 260, 'crashed': 31, 'out_of_time': 2, 'landed_off_pad': 5, 'flew_away': 2} |
| Final AI (start) | 142 | 3 | 72.7% (67%–77%) | 192.0 | 93 | 11.7% | {'landed': 218, 'hit_obstacle': 53, 'crashed': 24, 'landed_off_pad': 3, 'flew_away': 1, 'out_of_time': 1} |
| Final AI (start) | 142 | 4 | 3.3% (2%–6%) | 101.7 | 124 | 0.0% | {'hit_obstacle': 189, 'stuck_on_first_pad': 95, 'crashed': 5, 'landed': 10, 'out_of_time': 1} |
| Final AI (start) | 142 | 4a | 21.3% (17%–26%) | 219.4 | 148 | 0.0% | {'stuck_on_first_pad': 214, 'landed': 64, 'crashed': 16, 'out_of_time': 3, 'flew_away': 1, 'landed_off_pad': 2} |
| Level 1 brain | 142 | 1 | 90.3% (86%–93%) | 254.8 | 93 | 12.3% | {'landed': 271, 'crashed': 24, 'landed_off_pad': 5} |
| Level 2 brain | 142 | 1 | 95.3% (92%–97%) | 269.3 | 90 | 9.0% | {'landed': 286, 'crashed': 10, 'landed_off_pad': 3, 'flew_away': 1} |
| Level 2 brain | 142 | 2 | 87.7% (83%–91%) | 249.0 | 91 | 14.0% | {'landed': 263, 'crashed': 33, 'landed_off_pad': 2, 'flew_away': 1, 'out_of_time': 1} |
| Level 3 brain | 142 | 1 | 96.7% (94%–98%) | 271.7 | 89 | 3.3% | {'landed': 290, 'crashed': 7, 'out_of_time': 2, 'landed_off_pad': 1} |
| Level 3 brain | 142 | 2 | 93.0% (90%–95%) | 266.8 | 91 | 7.7% | {'landed': 279, 'crashed': 16, 'landed_off_pad': 4, 'out_of_time': 1} |
| Level 3 brain | 142 | 3 | 82.7% (78%–87%) | 233.5 | 91 | 13.0% | {'landed': 248, 'crashed': 28, 'hit_obstacle': 21, 'out_of_time': 3} |
| Mission 4a brain | 142 | 4a | 79.3% (74%–84%) | 377.1 | 171 | 0.3% | {'landed': 238, 'crashed': 12, 'stuck_on_first_pad': 19, 'landed_off_pad': 16, 'out_of_time': 13, 'flew_away': 2} |
| Mission 4a brain | 142 | 4 | 28.7% (24%–34%) | 170.0 | 168 | 0.3% | {'hit_obstacle': 185, 'out_of_time': 8, 'landed': 86, 'crashed': 9, 'stuck_on_first_pad': 6, 'landed_off_pad': 5, 'flew_away': 1} |
| Mission 4a brain, overtime | 142 | 4a | 87.3% (83%–91%) | 412.8 | 148 | 0.0% | {'landed': 262, 'crashed': 9, 'stuck_on_first_pad': 19, 'landed_off_pad': 5, 'out_of_time': 5} |
| Mission 4a brain, overtime | 142 | 4 | 29.7% (25%–35%) | 195.0 | 145 | 0.3% | {'hit_obstacle': 197, 'out_of_time': 4, 'landed': 89, 'crashed': 2, 'stuck_on_first_pad': 6, 'landed_off_pad': 2} |
| Final AI (start) | 242 | 1 | 74.0% (69%–79%) | 210.4 | 99 | 40.0% | {'landed': 222, 'crashed': 76, 'landed_off_pad': 2} |
| Final AI (start) | 242 | 2 | 68.7% (63%–74%) | 197.9 | 100 | 38.7% | {'landed': 206, 'crashed': 91, 'landed_off_pad': 3} |
| Final AI (start) | 242 | 3 | 58.3% (53%–64%) | 161.4 | 99 | 33.7% | {'landed': 175, 'hit_obstacle': 44, 'crashed': 80, 'landed_off_pad': 1} |
| Final AI (start) | 242 | 4 | 5.3% (3%–8%) | 117.6 | 161 | 0.0% | {'hit_obstacle': 196, 'stuck_on_first_pad': 77, 'crashed': 10, 'landed': 16, 'landed_off_pad': 1} |
| Final AI (start) | 242 | 4a | 29.7% (25%–35%) | 251.6 | 175 | 0.0% | {'stuck_on_first_pad': 180, 'landed': 89, 'out_of_time': 5, 'crashed': 16, 'landed_off_pad': 10} |
| Level 1 brain | 242 | 1 | 91.0% (87%–94%) | 258.8 | 91 | 15.0% | {'landed': 273, 'crashed': 25, 'landed_off_pad': 2} |
| Level 2 brain | 242 | 1 | 95.0% (92%–97%) | 266.7 | 91 | 8.7% | {'landed': 285, 'crashed': 14, 'landed_off_pad': 1} |
| Level 2 brain | 242 | 2 | 86.3% (82%–90%) | 248.3 | 92 | 17.3% | {'landed': 259, 'crashed': 36, 'landed_off_pad': 4, 'flew_away': 1} |
| Level 3 brain | 242 | 1 | 96.0% (93%–98%) | 273.4 | 89 | 5.3% | {'landed': 288, 'crashed': 8, 'landed_off_pad': 4} |
| Level 3 brain | 242 | 2 | 92.3% (89%–95%) | 267.0 | 91 | 11.3% | {'landed': 277, 'crashed': 18, 'landed_off_pad': 3, 'flew_away': 1, 'out_of_time': 1} |
| Level 3 brain | 242 | 3 | 85.7% (81%–89%) | 245.3 | 91 | 14.0% | {'landed': 257, 'crashed': 27, 'landed_off_pad': 1, 'hit_obstacle': 14, 'out_of_time': 1} |
| Mission 4a brain | 242 | 4a | 76.0% (71%–80%) | 390.3 | 148 | 0.0% | {'stuck_on_first_pad': 65, 'landed': 228, 'crashed': 4, 'out_of_time': 3} |
| Mission 4a brain | 242 | 4 | 22.0% (18%–27%) | 209.5 | 142 | 0.0% | {'hit_obstacle': 193, 'stuck_on_first_pad': 37, 'crashed': 2, 'landed': 66, 'landed_off_pad': 1, 'out_of_time': 1} |
| Mission 4a brain, overtime | 242 | 4a | 79.7% (75%–84%) | 391.6 | 159 | 0.3% | {'stuck_on_first_pad': 48, 'landed': 239, 'out_of_time': 5, 'crashed': 6, 'landed_off_pad': 2} |
| Mission 4a brain, overtime | 242 | 4 | 27.7% (23%–33%) | 206.5 | 155 | 0.0% | {'hit_obstacle': 199, 'landed': 83, 'crashed': 4, 'landed_off_pad': 4, 'stuck_on_first_pad': 8, 'out_of_time': 2} |
