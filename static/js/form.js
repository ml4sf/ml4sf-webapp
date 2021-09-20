$("#theForm").submit(function (e) {
    e.preventDefault();
});

$("#btnProcess").on('click', function (e) {
    var self = this;
    $.ajax({
        type: "POST",
        url: "/process",
        data: new FormData(document.getElementById('theForm')),
        cache: false,
        contentType: false,
        processData: false,
        beforeSend: () => {
            $(self).html(
                '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>'
            );
            $(self).prop("disabled", true);
            $("#results").html("");
        }
    })
    .done((data) => $("#results").html(data))

    .fail(() => $("#results").html("Please try again"))

    .always(() => {
        $(self).html('Process');
        $(self).removeAttr('disabled');
    });
});

$("#btnSubmit").on('click', function (e) {
    e.preventDefault();
    var subject = $("#txtSubject").val();
    var body = $("#txtMsg").val();
    window.location.href="mailto:jromanova@chem.uni-sofia.bg?subject=" + subject + "&body=" + body
    $("#mailLink").attr('href',"mailto:jromanova@chem.uni-sofia.bg?subject=" + subject + "&body=" + body);
    $("#mailLink").click();
});
