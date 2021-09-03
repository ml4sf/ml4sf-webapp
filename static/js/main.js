(function ($) {
    'use strict';
    //==========preloder===========
    $(window).on('load',function(){
        var preLoder = $(".preloader");
        preLoder.fadeOut(1000);
    });
    //==========fixed header & scroll to top button===========
    $(window).on('scroll', function(){
        if($(window).scrollTop() > 300) {
            $('.bottom-header').addClass('animated fadeInDown fixed-header');
            // scroll to top button show
            $('.scroll-to-top').fadeIn();
            $('.scroll-to-top').addClass('active');
            if ($(window).width() < 992) {
                $('.bottom-header').removeClass('animated fadeInDown fixed-header');
            }
        } else {
            $('.bottom-header').removeClass('animated fadeInDown fixed-header');
            // scroll to top button show
            $('.scroll-to-top').fadeOut();
            $('.scroll-to-top').removeClass('active');
        }
    });
    $(document).ready(function(){
        // scroll to top function
        $(".scroll-to-top").on('click', function(){
            $('html, body').animate({scrollTop : 0},100);
        })
        //==========modal video popup===========
        $(".vdo-btn").modalVideo();
        //==========portfolio slider===========
        $('.items').owlCarousel({
            loop: true,
            smartSpeed: 1000,
            margin: 30,
            autoplay: true,
            autoplayTimeout: 5000,
            autoplayHoverPause: false,
            dots: false,
            nav: true,
            navText: ['<i class="fas fa-arrow-left"></i>','<i class="fas fa-arrow-right"></i>'],
            responsive: {
                0: {
                    items: 1
                },
                768: {
                    items: 2
                },
                960: {
                    items: 3
                },
                1200: {
                    items: 3
                },
                1920: {
                    items: 3
                }
            }
        });
        // odometer
        $('.odometer').appear(function(e) {
            var odo = $(".odometer");
            odo.each(function() {
                var countNumber = $(this).attr("data-count");
                $(this).html(countNumber);
            });
        });
        //==========testimonial slider===========
        $('.comments').slick({
            slidesToShow: 1,
            slidesToScroll: 1,
            arrows: false,
            fade: true,
            asNavFor: '.clients'
        });
        $('.clients').slick({
            slidesToShow: 3,
            slidesToScroll: 1,
            asNavFor: '.comments',
            centerMode: true,
            centerPadding : 0,
            autoplay : true,
            prevArrow : '<button type="button" class="slick-prev"><i class="fas fa-arrow-left"></i></button>',
            nextArrow : '<button type="button" class="slick-next"><i class="fas fa-arrow-right"></i></button>',
            responsive: [
                {
                    breakpoint: 600,
                    settings: {
                        arrows : false
                    }
                },
                {
                    breakpoint: 480,
                    settings: {
                        arrows : false
                    }
                }
            ]
        });
        // faq section collapse
        $('.collapse').on('shown.bs.collapse', function(){
            $(this).parent().find('.fa-plus').removeClass('fa-plus').addClass('fa-minus');
        }).on('hidden.bs.collapse', function(){
            $(this).parent().find('.fa-minus').removeClass('fa-minus').addClass('fa-plus');
        })
        //==========brands slider===========
        $('.all-brand').slick({
            arrows: false,
            slidesToShow: 5,
            slidesToScroll: 1,
            autoplay : true,
            speed: 1500,
            autoplaySpeed: 2000,
            responsive: [
                {
                    breakpoint: 1024,
                    settings: {
                        slidesToShow: 4
                    }
                },
                {
                    breakpoint: 600,
                    settings: {
                        slidesToShow: 3
                    }
                },
                {
                    breakpoint: 480,
                    settings: {
                        slidesToShow: 2
                    }
                }
            ]
        });
        //==========inner page testimonial slider===========
        $('.comment-box').owlCarousel({
            loop: true,
            smartSpeed: 1000,
            margin: 5,
            autoplay: true,
            autoplayTimeout: 5000,
            autoplayHoverPause: false,
            dots: false,
            responsive: {
                0: {
                    items: 1
                },
                768: {
                    items: 1
                },
                960: {
                    items: 1
                },
                1200: {
                    items: 1
                },
                1920: {
                    items: 1
                }
            }
        });
        //==========project page main slider===========
        $('.part-images').owlCarousel({
            loop: true,
            smartSpeed: 1000,
            margin: 15,
            autoplay: true,
            autoplayTimeout: 5000,
            autoplayHoverPause: false,
            dots: false,
            nav: true,
            navText: ['<i class="fas fa-arrow-left"></i>','<i class="fas fa-arrow-right"></i>'],
            responsive: {
                0: {
                    items: 1
                },
                768: {
                    items: 1
                },
                960: {
                    items: 1
                },
                1200: {
                    items: 1
                },
                1920: {
                    items: 1
                }
            }
        });
    });
}(jQuery));