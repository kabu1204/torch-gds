#include <unordered_set>

/*
 * Data pointers of tensors to be pinned to Direct-Host-Access
*/
extern std::unordered_set<void*> ToDHATensorSet;
/*
 * Data pointers of tensors already pinned to Direct-Host-Access
*/
extern std::unordered_set<void*> DHATensorSet;