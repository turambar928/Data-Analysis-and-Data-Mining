def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)

    # Create a DP table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize DP table
    for i in range(m + 1):
        dp[i][0] = i  # Minimum operations = delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Minimum operations = insert all characters

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j],  # Delete
                    dp[i - 1][j - 1]  # Substitute
                )

    return dp[m][n]


# Test cases
test_cases = [
    ("goodgoodStudy", ""),
    ("", "daydayCode"),
    ("intention", "execution"),
    ("kitten", "sitting")  # Additional example from the prompt
]

for str1, str2 in test_cases:
    print(f"str1 = '{str1}', str2 = '{str2}' => Edit distance: {edit_distance(str1, str2)}")