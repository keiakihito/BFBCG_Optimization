#ifndef LARGEVECTOR_HPP
#define LARGEVECTOR_HPP

#include <glm/glm.hpp>
#include <vector>
#include <cstring>

template<class T>
class LargeVector {
private:
	std::vector<T> v;

public:

	LargeVector() {

	}
	LargeVector(const LargeVector& other) {
		v.resize(other.v.size());
		memcpy(&v[0], &(other.v[0]), sizeof(other.v[0])*other.v.size());
	}
	void resize(const int size) {
		v.resize(size);
	}
	void clear(bool isIdentity=false) {
		memset(&v[0], 0, sizeof(T)*v.size());
		if(isIdentity) {
			for(size_t i=0;i<v.size();i++) {
				v[i] = T(1);
			}
		}
	}

	size_t size() {
		return v.size();
	}

	size_t size() const {
		return v.size();
	}


	T& operator[](size_t index) {
		return v[index];
	}

	const T& operator[](size_t index) const{
		return v[index];
	}


	friend LargeVector<glm::vec3> operator*(const LargeVector<glm::mat3> other, const LargeVector<glm::vec3> f );
	friend LargeVector<glm::vec3> operator*(const float f, const LargeVector<glm::vec3> other);
	friend LargeVector<glm::vec3> operator-(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb );
	//friend LargeVector<T> operator+(const LargeVector<T> Va, const LargeVector<T> Vb );
	friend LargeVector<glm::vec3> operator+(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb );

	friend LargeVector<glm::mat3> operator*(const float f, const LargeVector<glm::mat3> other);
	friend LargeVector<glm::mat3> operator-(const LargeVector<glm::mat3> Va, const LargeVector<glm::mat3> Vb );
	//friend LargeVector<glm::mat3> operator+(const LargeVector<glm::mat3> Va, const LargeVector<glm::mat3> Vb );


	friend LargeVector<glm::vec3> operator/(const float f, const LargeVector<glm::vec3> v );
	friend float dot(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb );
};

LargeVector<glm::vec3> operator*(const LargeVector<glm::mat3> other, const LargeVector<glm::vec3> v ) {
	LargeVector<glm::vec3> tmp(v);
	for(size_t i=0;i<v.v.size();i++) {
		tmp.v[i] = other.v[i] * v.v[i];
	}
	return tmp;
}

LargeVector<glm::vec3> operator*(const float f, const LargeVector<glm::vec3> other) {
	LargeVector<glm::vec3> tmp(other);
	for(size_t i=0;i<other.v.size();i++) {
		tmp.v[i] = other.v[i]*f;
	}
	return tmp;
}
LargeVector<glm::mat3> operator*(const float f, const LargeVector<glm::mat3> other) {
	LargeVector<glm::mat3> tmp(other);
	for(size_t i=0;i<other.v.size();i++) {
		tmp.v[i] = other.v[i]*f;
	}
	return tmp;
}
LargeVector<glm::vec3> operator-(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb ) {
	LargeVector<glm::vec3> tmp(Va);
	for(size_t i=0;i<Va.v.size();i++) {
		tmp.v[i] = Va.v[i] - Vb.v[i];
	}
	return tmp;
}
LargeVector<glm::mat3> operator-(const LargeVector<glm::mat3> Va, const LargeVector<glm::mat3> Vb ) {
	LargeVector<glm::mat3> tmp(Va);
	for(size_t i=0;i<Va.v.size();i++) {
		tmp.v[i] = Va.v[i] - Vb.v[i];
	}
	return tmp;
}

LargeVector<glm::vec3> operator+(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb ) {
	LargeVector<glm::vec3> tmp(Va);
	for(size_t i=0;i<Va.v.size();i++) {
		tmp.v[i] = Va.v[i] + Vb.v[i];
	}
	return tmp;
}

LargeVector<glm::vec3> operator/(const float f, const LargeVector<glm::vec3> v ) {
	LargeVector<glm::vec3> tmp(v);
	for(size_t i=0;i<v.v.size();i++) {
		tmp.v[i] = v.v[i] / f;
	}
	return tmp;
}



#endif // LARGEVECTOR_HPP
