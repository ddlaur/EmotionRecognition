### Question: How to change the website language once a button is clicked?

I am new to HTML and I am trying to figure out how to change the website language once a button is clicked. For example, I created a website with 2 flags (US, ES) and I want the whole page to change the language once the flag is clicked.

However, I'm not sure how to do it since it seems to me like all of the text I used is static, but I'm not sure how to do it otherwise.

For example, my code looks like this:

```html
<header id="header" class="row">   
    <nav id="header-nav-wrap">
        <ul class="header-main-nav">
            <li class="current"><a class="smoothscroll" href="#home" title="home">Home</a></li>
            <li><a class="smoothscroll" href="#about" title="about">About</a></li>
            <li><a class="smoothscroll" href="#download" title="download">Download</a></li>
        </ul>
    </nav>
    
    <img class="esp" href="#esp" src="https://www.countryflags.io/es/shiny/64.png" height="30" width="30" onclick="document.body.className='es'">
    <img class="english" href="#english" src="https://www.countryflags.io/us/shiny/64.png" height="30" width="30" onclick="document.body.className='en'">

    <a class="header-menu-toggle" href="#"><span>Menu</span></a>        
</header> ```

Now here the Home/About/Download are static strings from my understanding. How can I change them once clicked?

---

### Answer

Handling Internationalization is not a small task. There are many pieces that you need to prepare before you can properly supply language-specific page content.

That being said, the `lang` global attribute defines the language of an element. Adding this step to the work done in your click event would be a good idea. Using `Element.setAttribute()` you can update the `lang` attribute on the `<body>` and update nav link text with `Node.textContent` every time the locale flags are pressed.

**Note:** The default value of `lang` is unknown, therefore it is recommended to always specify this attribute with the appropriate value.

There are a few ways you could go about doing this. I see you are adding the locale as a class to the `<body>` inside the inline `onclick` event. If you wanted to change the nav link text based on the body having a class of "en" or "esp" you could do that. Or you could attach event listeners to each of the `<img>` elements and then update the nav link text and `lang` depending on which locale flag is clicked.

You will need to swap the current text with a locale-specific type on each click event, so I added some demo data to represent the `en-US` and `es` content.

> Now here the Home/About/Download are static strings from my understanding. How can I change them once clicked?

The text content of each `<a>` element in your list of nav links is indeed static text, but when a click event is triggered from the `<img>` flag being pressed, we can dynamically update the `textContent` for those `<a>` nodes on the DOM using `Node.textContent` and reassign its value like `Node.textContent = ""`.

**JavaScript:**

```javascript
const espFlag = document.querySelector(".esp");
const engFlag = document.querySelector(".english");
const navLinks = document.querySelectorAll(".smoothscroll");
const body = document.querySelector("body");

const espLocaleLinks = ["One", "Two", "Three"];
const engLocaleLinks = ["Home", "About", "Download"];

espFlag.addEventListener("click", () => {
   body.setAttribute("lang", "es");
   body.className = 'es';
   navLinks.forEach((link, index) => link.textContent = espLocaleLinks[index]);
   console.log(body);
});

engFlag.addEventListener("click", () => {
   body.setAttribute("lang", "en-US");
   body.className = 'en';
   navLinks.forEach((link, index) => link.textContent = engLocaleLinks[index]);
   console.log(body);
});

```

**HTML:**

```html
<header id="header" class="row">   
    <nav id="header-nav-wrap">
        <ul class="header-main-nav">
            <li class="current"><a class="smoothscroll" href="#home" title="home">Home</a></li>
            <li><a class="smoothscroll" href="#about" title="about">About</a></li>
            <li><a class="smoothscroll" href="#download" title="download">Download</a></li>
        </ul>
    </nav>
    
    <img class="esp" href="#esp" src="https://www.countryflags.io/es/shiny/64.png" height="30" width="30">
    <img class="english" href="#english" src="https://www.countryflags.io/us/shiny/64.png" height="30" width="30">

    <a class="header-menu-toggle" href="#"><span>Menu</span></a>        
</header> ```

```