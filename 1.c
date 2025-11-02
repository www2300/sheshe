#include <stdio.h>
#include <string.h>

int calculate_score(const char* str1, const char* str2) {
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    int max_possible = (len1 < len2) ? len1 : len2;

    for (int k = max_possible; k > 0; k--) {
        int match = 1;
        for (int i = 0; i < k; i++) {
            if (str1[len1 - k + i] != str2[i]) {
                match = 0;
                break;
            }
        }
        if (match) return k;
    }
    return 0;
}

int main() {
    char s[1005], t[1005];
    printf("请输入两个字符串（分别回车）：\n");

    if (scanf("%1000s", s) != 1) {
        printf("输入第一个字符串失败\n");
        return 0;
    }
    if (scanf("%1000s", t) != 1) {
        printf("输入第二个字符串失败\n");
        return 0;
    }

    int score1 = calculate_score(s, t);
    int score2 = calculate_score(t, s);
    int min_score = (score1 < score2) ? score1 : score2;

    printf("最小得分: %d\n", min_score);
    return 0;
}
