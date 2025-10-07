const apikey="7f89e385a9d44fcff8d1c8fc93d0179c";
const apiurl="https://api.openweathermap.org/data/2.5/weather?units=metric&q=";
const searchBox= document.querySelector(".search input");
const searchBtn= document.querySelector(".search button");
const weatherIcon=document.querySelector(".weather-icon");


async function checkWeather(city) {
    const response = await fetch(apiurl+city+`&appid=${apikey}`);
    
    if(response.status== 404){
        document.querySelector(".error").style.display="block";
        document.querySelector(".weather").style.display="none";
    }
    else{
    var data = await response.json();

    console.log(data);
    document.querySelector(".city").innerHTML = data.name;
    document.querySelector(".temp").innerHTML = Math.round(data.main.temp) + "°C" ;
    document.querySelector(".humidity").innerHTML = data.main.humidity + "%";
    document.querySelector(".wind").innerHTML = data.wind.speed + "km/h";
    if(data.weather[0].main=="Clouds"){
        weatherIcon.src = "images/clouds.png"
    }
    else if(data.weather[0].main=="Clear"){
        weatherIcon.src = "images/clear.png"
    }
    else if(data.weather[0].main=="Rain"){
        weatherIcon.src = "images/rain.png"
    }
    else if(data.weather[0].main=="Drizzle"){
        weatherIcon.src = "images/drizzle.png"
    }
    else if(data.weather[0].main=="Mist"){
        weatherIcon.src = "images/mist.png"
    }
document.querySelector(".weather").style.display = "block";
document.querySelector(".error").style.display="none";
}
}
searchBtn.addEventListener("click", ()=>{
checkWeather(searchBox.value);
})
searchBox.addEventListener("keydown", (event)=>{
  if(event.key==='Enter')  checkWeather(searchBox.value);
    })

        const starryNight = document.querySelector('.starry-night');
        const starCount = 100;
        for (let i = 0; i < starCount; i++) {
            const star = document.createElement('div');
            star.classList.add('star');
            star.style.top = `${Math.random() * 100}%`;
            star.style.left = `${Math.random() * 100}%`;
            star.style.animationDelay = `${Math.random() * 5}s`;
            starryNight.appendChild(star);
        }
