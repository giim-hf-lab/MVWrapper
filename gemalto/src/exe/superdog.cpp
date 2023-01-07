#ifdef __GEMALTO_USE_MODULE__

import gemalto;

#else

#include <gemalto/superdog.hpp>

#endif

int main()
{
	gemalto::superdog superdog;
	return not superdog.open() or not superdog.check_time();
}
