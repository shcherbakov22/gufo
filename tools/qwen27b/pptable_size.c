#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "smu14_driver_if_v14_0.h"
int main(void) {
  printf("sizeof(PPTable_t)        = %zu\n", sizeof(PPTable_t));
  printf("sizeof(PFE_Settings_t)   = %zu\n", sizeof(PFE_Settings_t));
  printf("sizeof(SkuTable_t)       = %zu\n", sizeof(SkuTable_t));
  printf("sizeof(CustomSkuTable_t) = %zu\n", sizeof(CustomSkuTable_t));
  printf("sizeof(BoardTable_t)     = %zu\n", sizeof(BoardTable_t));
  printf("TDC_THROTTLER_COUNT      = %d\n", TDC_THROTTLER_COUNT);
  printf("PPT_THROTTLER_COUNT      = %d\n", PPT_THROTTLER_COUNT);
  printf("offsets: PFE=%zu Sku=%zu CustomSku=%zu Board=%zu\n",
         offsetof(PPTable_t, PFE_Settings), offsetof(PPTable_t, SkuTable),
         offsetof(PPTable_t, CustomSkuTable), offsetof(PPTable_t, BoardTable));
  return 0;
}