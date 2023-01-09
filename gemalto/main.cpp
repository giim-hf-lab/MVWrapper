import gemalto;

int main()
{
	gemalto::superdog superdog;
	return not superdog.open() or not superdog.check_time();
}
